"""
Walk-forward portfolio backtesting engine.

Given a universe of symbols, an initial capital and one or more :class:`Strategy`
objects, this simulates investing over a historical window with periodic
rebalancing and produces an equity curve plus risk/return metrics per strategy.

Design notes:
- **No look-ahead.** At each rebalance date the strategy is handed price history
  only up to and including that date; trades execute at that date's close.
- **Offline.** The engine reads prices exclusively from an injected
  :class:`MarketDataStore`. FX conversion and benchmark data are optional and
  injected, so the core simulation is fully unit-testable without network access.
- **Fractional shares** are allowed (the rest of the system already uses Decimal
  quantities), which avoids integer-rounding drift in the equity curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from .market_data_store import MarketDataStore
from .models import Currency
from .strategies import Strategy

logger = logging.getLogger(__name__)

_VALID_FREQUENCIES = {"daily", "weekly", "monthly", "quarterly", "none"}


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    symbols: List[str]
    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    rebalance_frequency: str = "monthly"  # daily|weekly|monthly|quarterly|none
    transaction_cost_bps: float = 0.0
    benchmark_symbol: str = "SPY"
    base_currency: Currency = Currency.USD
    risk_free_rate: float = 0.04
    lookback_days: int = 252  # history warm-up before start_date for strategies

    def __post_init__(self):
        self.symbols = [s.upper().strip() for s in self.symbols if s and s.strip()]
        self.rebalance_frequency = self.rebalance_frequency.lower().strip()
        if self.rebalance_frequency not in _VALID_FREQUENCIES:
            raise ValueError(
                f"rebalance_frequency must be one of {sorted(_VALID_FREQUENCIES)}"
            )
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if not self.symbols:
            raise ValueError("At least one symbol is required")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")


@dataclass
class StrategyResult:
    """Results for a single strategy within a backtest."""

    name: str
    equity_curve: pd.DataFrame  # date index, column "total_value"
    weights_history: List[Dict] = field(default_factory=list)  # {date, weights}
    trades: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    final_value: float = 0.0


@dataclass
class BacktestResult:
    """Aggregate result of a backtest across all strategies."""

    config: BacktestConfig
    strategies: List[StrategyResult]
    benchmark_curve: Optional[pd.DataFrame] = None
    price_start: Optional[date] = None
    price_end: Optional[date] = None
    warnings: List[str] = field(default_factory=list)


class Backtester:
    """Simulates strategies over historical prices from a MarketDataStore."""

    def __init__(
        self,
        market_data_store: MarketDataStore,
        metrics_calculator=None,
        fx_rate_on: Optional[
            Callable[[date, Currency, Currency], Optional[float]]
        ] = None,
        price_currencies: Optional[Dict[str, Currency]] = None,
    ):
        """
        Args:
            market_data_store: Source of historical prices.
            metrics_calculator: Optional ``FinancialMetricsCalculator`` used to
                compute performance metrics and benchmark data. If ``None``,
                metrics are skipped (useful for offline unit tests).
            fx_rate_on: Optional ``(date, from_ccy, to_ccy) -> rate`` callable used
                to convert non-base-currency price series to the base currency.
            price_currencies: Optional ``symbol -> Currency`` map giving the native
                currency of each symbol's stored prices. Only needed alongside
                ``fx_rate_on`` for multi-currency universes.
        """
        self.market_data_store = market_data_store
        self.metrics_calculator = metrics_calculator
        self.fx_rate_on = fx_rate_on
        self.price_currencies = price_currencies or {}

    # -- Public API ---------------------------------------------------------

    def run(self, config: BacktestConfig, strategies: List[Strategy]) -> BacktestResult:
        """Run one or more strategies over the configured window."""
        if not strategies:
            raise ValueError("At least one strategy is required")

        warnings: List[str] = []

        # Load prices with a warm-up buffer so strategies have trailing history
        # available on the very first rebalance date.
        buffer_start = config.start_date - timedelta(days=config.lookback_days + 10)
        prices = self.market_data_store.get_price_matrix(
            config.symbols, buffer_start, config.end_date
        )

        if prices.empty:
            raise ValueError(
                "No price data available for the requested symbols/date range. "
                "Update market data for this window first."
            )

        prices = self._to_base_currency(prices, config, warnings)
        prices = prices.dropna(axis=1, how="all")

        missing = [s for s in config.symbols if s not in prices.columns]
        if missing:
            warnings.append(f"No price data for: {', '.join(missing)} (excluded)")

        if prices.shape[1] == 0:
            raise ValueError("No usable price series for any requested symbol")

        # Simulation window = rows within [start, end].
        start_ts = pd.Timestamp(config.start_date)
        end_ts = pd.Timestamp(config.end_date)
        sim_mask = (prices.index >= start_ts) & (prices.index <= end_ts)
        sim_index = prices.index[sim_mask]
        if len(sim_index) < 2:
            raise ValueError(
                "Fewer than 2 data points in the backtest window; "
                "widen the date range or update market data."
            )

        rebalance_dates = self._rebalance_dates(sim_index, config.rebalance_frequency)

        results = [
            self._run_strategy(strategy, config, prices, sim_index, rebalance_dates)
            for strategy in strategies
        ]

        benchmark_curve = self._build_benchmark_curve(config, sim_index)

        return BacktestResult(
            config=config,
            strategies=results,
            benchmark_curve=benchmark_curve,
            price_start=sim_index[0].date(),
            price_end=sim_index[-1].date(),
            warnings=warnings,
        )

    # -- Core simulation ----------------------------------------------------

    def _run_strategy(
        self,
        strategy: Strategy,
        config: BacktestConfig,
        prices: pd.DataFrame,
        sim_index: pd.DatetimeIndex,
        rebalance_dates: set,
    ) -> StrategyResult:
        symbols = list(prices.columns)
        cash = float(config.initial_capital)
        holdings: Dict[str, float] = {s: 0.0 for s in symbols}
        cost_rate = config.transaction_cost_bps / 10_000.0

        equity_dates: List[pd.Timestamp] = []
        equity_values: List[float] = []
        weights_history: List[Dict] = []
        trades: List[Dict] = []

        for ts in sim_index:
            day_prices = prices.loc[ts]
            tradeable = [s for s in symbols if pd.notna(day_prices[s])]

            if ts.date() in rebalance_dates and tradeable:
                # Value the book at today's close (skip untradeable holdings).
                current_value = cash + sum(
                    holdings[s] * float(day_prices[s]) for s in tradeable
                )
                current_weights = {
                    s: (
                        (holdings[s] * float(day_prices[s]) / current_value)
                        if current_value > 0
                        else 0.0
                    )
                    for s in tradeable
                }

                # Strategy only ever sees data up to and including ``ts``.
                window = prices.loc[:ts, tradeable].dropna(how="all")
                target = strategy.target_weights(ts.date(), window, current_weights)

                if target:
                    cash, day_trades = self._rebalance(
                        ts,
                        target,
                        tradeable,
                        day_prices,
                        holdings,
                        cash,
                        current_value,
                        cost_rate,
                    )
                    trades.extend(day_trades)
                    weights_history.append(
                        {
                            "date": ts.date().isoformat(),
                            "weights": {
                                s: round(w, 6)
                                for s, w in self._normalize(target, tradeable).items()
                            },
                        }
                    )

            # End-of-day valuation (holdings valued at last known price via ffill).
            value = cash + sum(
                holdings[s] * float(day_prices[s])
                for s in symbols
                if pd.notna(day_prices[s])
            )
            equity_dates.append(ts)
            equity_values.append(value)

        equity_curve = pd.DataFrame(
            {"total_value": equity_values}, index=pd.DatetimeIndex(equity_dates)
        )

        metrics = {}
        if self.metrics_calculator is not None:
            try:
                metrics = self.metrics_calculator.calculate_metrics_from_df(
                    equity_curve,
                    value_column="total_value",
                    benchmark_symbol=config.benchmark_symbol,
                    risk_free_rate=config.risk_free_rate,
                )
            except Exception as e:
                logger.warning(f"Metrics calculation failed for {strategy.name}: {e}")
                metrics = {"error": str(e)}

        return StrategyResult(
            name=strategy.name,
            equity_curve=equity_curve,
            weights_history=weights_history,
            trades=trades,
            metrics=metrics,
            final_value=equity_values[-1] if equity_values else 0.0,
        )

    def _rebalance(
        self,
        ts,
        target,
        tradeable,
        day_prices,
        holdings,
        cash,
        portfolio_value,
        cost_rate,
    ):
        """Move holdings toward ``target`` weights at today's close; returns
        (new_cash, trades). Mutates ``holdings`` in place."""
        norm = self._normalize(target, tradeable)
        trades: List[Dict] = []

        for s in tradeable:
            price = float(day_prices[s])
            if price <= 0:
                continue
            target_value = portfolio_value * norm.get(s, 0.0)
            target_shares = target_value / price
            delta_shares = target_shares - holdings[s]
            notional = delta_shares * price
            if abs(notional) < 1e-9:
                continue

            fee = abs(notional) * cost_rate
            cash -= notional  # buying (delta>0) spends cash; selling adds cash
            cash -= fee
            holdings[s] = target_shares

            trades.append(
                {
                    "date": ts.date().isoformat(),
                    "symbol": s,
                    "action": "BUY" if delta_shares > 0 else "SELL",
                    "shares": round(abs(delta_shares), 6),
                    "price": round(price, 4),
                    "notional": round(abs(notional), 2),
                    "fee": round(fee, 4),
                }
            )

        return cash, trades

    @staticmethod
    def _normalize(weights: Dict[str, float], tradeable: List[str]) -> Dict[str, float]:
        """Restrict to tradeable symbols and renormalize to sum to 1.0."""
        restricted = {s: w for s, w in weights.items() if s in tradeable and w > 0}
        total = sum(restricted.values())
        if total <= 0:
            return {}
        return {s: w / total for s, w in restricted.items()}

    # -- Helpers ------------------------------------------------------------

    def _rebalance_dates(self, sim_index: pd.DatetimeIndex, freq: str) -> set:
        """Return the set of dates on which to rebalance.

        The first simulation date is always included (initial deployment).
        """
        dates = {sim_index[0].date()}
        if freq in ("none",):
            return dates
        if freq == "daily":
            return {ts.date() for ts in sim_index}

        def period_key(ts: pd.Timestamp):
            if freq == "weekly":
                iso = ts.isocalendar()
                return (iso[0], iso[1])
            if freq == "monthly":
                return (ts.year, ts.month)
            if freq == "quarterly":
                return (ts.year, (ts.month - 1) // 3)
            return None

        last_key = period_key(sim_index[0])
        for ts in sim_index[1:]:
            key = period_key(ts)
            if key != last_key:
                dates.add(ts.date())  # first trading day of the new period
                last_key = key
        return dates

    def _to_base_currency(
        self, prices: pd.DataFrame, config: BacktestConfig, warnings: List[str]
    ) -> pd.DataFrame:
        """Convert each price series to the base currency using historical FX.

        No-op when no FX callable / currency map is provided (assumes prices are
        already in the base currency — the common single-currency case).
        """
        if self.fx_rate_on is None or not self.price_currencies:
            return prices

        base = config.base_currency
        converted = {}
        for symbol in prices.columns:
            native = self.price_currencies.get(symbol, base)
            series = prices[symbol]
            if native == base:
                converted[symbol] = series
                continue

            rate_cache: Dict[date, Optional[float]] = {}
            new_vals = {}
            for ts, price in series.items():
                if pd.isna(price):
                    continue
                d = ts.date()
                if d not in rate_cache:
                    rate_cache[d] = self.fx_rate_on(d, native, base)
                rate = rate_cache[d]
                if rate:
                    new_vals[ts] = float(price) * float(rate)
            if new_vals:
                converted[symbol] = pd.Series(new_vals)
            else:
                warnings.append(
                    f"Could not convert {symbol} from {native.value} to {base.value}; excluded"
                )

        if not converted:
            return pd.DataFrame(index=prices.index)
        result = pd.DataFrame(converted).sort_index().ffill()
        return result

    def _build_benchmark_curve(
        self, config: BacktestConfig, sim_index: pd.DatetimeIndex
    ) -> Optional[pd.DataFrame]:
        """Build a normalized benchmark equity curve (starts at initial_capital)."""
        if self.metrics_calculator is None or not config.benchmark_symbol:
            return None
        try:
            _, benchmark_prices = self.metrics_calculator.get_benchmark_data(
                config.benchmark_symbol, sim_index[0].date(), sim_index[-1].date()
            )
        except Exception as e:
            logger.warning(f"Could not fetch benchmark {config.benchmark_symbol}: {e}")
            return None

        if not benchmark_prices:
            return None

        series = pd.Series(
            {pd.Timestamp(d): float(p) for d, p in benchmark_prices.items()}
        ).sort_index()
        if series.empty:
            return None

        base_price = series.iloc[0]
        if base_price <= 0:
            return None
        equity = series / base_price * config.initial_capital
        return pd.DataFrame({"total_value": equity})
