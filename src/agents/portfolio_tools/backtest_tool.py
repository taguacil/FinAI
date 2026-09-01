"""
Backtest tool: run walk-forward strategy backtests over historical prices.

Thin orchestration layer over ``src.portfolio.backtester`` for use from the MCP
server (and, indirectly, the AI agent). It resolves the universe (explicit symbols
or the current portfolio's positions), optionally warms up market data for the
window, runs one or more strategies, and formats a comparison summary as text.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from ...portfolio.backtester import BacktestConfig, Backtester
from ...portfolio.manager import PortfolioManager
from ...portfolio.models import Currency
from ...portfolio.strategies import build_strategies

logger = logging.getLogger(__name__)

DEFAULT_STRATEGIES = "hrp,equal_weight,buy_and_hold"


class BacktestPortfolioTool:
    """Backtest one or more strategies over a historical window."""

    name = "backtest_portfolio"

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        data_manager,
        metrics_calculator=None,
    ):
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager
        self.metrics_calculator = metrics_calculator

    # -- universe resolution ------------------------------------------------

    def _universe_from_portfolio(self) -> Tuple[List[str], Dict[str, Currency]]:
        """Resolve (price-symbols, currency-map) from the current portfolio.

        Uses each position's effective price symbol (data_provider_symbol or
        symbol) so lookups hit the same keys market data was stored under.
        """
        portfolio = self.portfolio_manager.current_portfolio
        symbols: List[str] = []
        currencies: Dict[str, Currency] = {}
        if not portfolio:
            return symbols, currencies

        for sym, pos in portfolio.positions.items():
            if pos.quantity <= 0:
                continue
            inst = pos.instrument
            price_symbol = (inst.data_provider_symbol or inst.symbol) if inst else sym
            price_symbol = price_symbol.upper().strip()
            symbols.append(price_symbol)
            if inst:
                currencies[price_symbol] = inst.price_currency or inst.currency
        return symbols, currencies

    def _currency_map_from_portfolio(self) -> Dict[str, Currency]:
        """Map every known price symbol (from positions) → its price currency.

        Includes zero-quantity positions so an explicitly-backtested symbol that
        the user isn't currently holding can still be currency-resolved.
        """
        portfolio = self.portfolio_manager.current_portfolio
        currencies: Dict[str, Currency] = {}
        if not portfolio:
            return currencies
        for sym, pos in portfolio.positions.items():
            inst = pos.instrument
            price_symbol = (inst.data_provider_symbol or inst.symbol) if inst else sym
            price_symbol = price_symbol.upper().strip()
            if inst:
                currencies[price_symbol] = inst.price_currency or inst.currency
        return currencies

    # -- main entry ---------------------------------------------------------

    def _run(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[str] = None,
        strategies: str = DEFAULT_STRATEGIES,
        initial_capital: float = 100_000.0,
        rebalance_frequency: str = "monthly",
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
        benchmark: str = "SPY",
        transaction_cost_bps: float = 0.0,
        ensure_data: bool = True,
    ) -> str:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            return "❌ Dates must be in YYYY-MM-DD format."

        portfolio = self.portfolio_manager.current_portfolio
        base_currency = portfolio.base_currency if portfolio else Currency.USD

        # Resolve universe: explicit symbols, else current portfolio positions.
        price_currencies: Dict[str, Currency] = {}
        fx_warnings: List[str] = []
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            # Try to resolve each explicit symbol's price currency from the
            # current portfolio so FX conversion still applies. Symbols we can't
            # map are assumed to already be in the base currency; warn so a
            # foreign-listed instrument isn't silently treated as base.
            known = self._currency_map_from_portfolio()
            unresolved: List[str] = []
            for sym in symbol_list:
                if sym in known:
                    price_currencies[sym] = known[sym]
                elif sym == benchmark.upper().strip():
                    continue  # benchmark handled separately
                else:
                    unresolved.append(sym)
            if unresolved:
                fx_warnings.append(
                    "No currency info for "
                    + ", ".join(unresolved)
                    + f"; assuming prices are already in {base_currency.value} "
                    "(FX conversion not applied)."
                )
        else:
            symbol_list, price_currencies = self._universe_from_portfolio()
            if not symbol_list:
                return (
                    "❌ No symbols provided and no positions in the current portfolio. "
                    "Pass symbols='AAPL,MSFT,...' or load a portfolio first."
                )

        if len(symbol_list) < 1:
            return "❌ Need at least one symbol to backtest."

        strategy_specs = [s.strip() for s in strategies.split(",") if s.strip()]
        if not strategy_specs:
            return "❌ No strategies specified."

        try:
            strat_objs = build_strategies(
                strategy_specs,
                lookback_days=lookback_days,
                risk_free_rate=risk_free_rate,
            )
        except ValueError as e:
            return f"❌ {e}"

        # Optionally warm up market data (network) for the window + lookback buffer.
        if ensure_data:
            try:
                from datetime import timedelta

                self.portfolio_manager.update_market_data(
                    start - timedelta(days=lookback_days + 10), end
                )
                self.portfolio_manager.market_data_store.clear_cache()
            except Exception as e:
                logger.warning(f"Could not warm up market data: {e}")

        # FX conversion only when the universe has non-base currencies.
        fx_rate_on = None
        if any(c != base_currency for c in price_currencies.values()):

            def fx_rate_on(d: date, frm: Currency, to: Currency):
                try:
                    rate = self.data_manager.get_historical_fx_rate_on(d, frm, to)
                    return float(rate) if rate else None
                except Exception:
                    return None

        try:
            config = BacktestConfig(
                symbols=symbol_list,
                start_date=start,
                end_date=end,
                initial_capital=initial_capital,
                rebalance_frequency=rebalance_frequency,
                transaction_cost_bps=transaction_cost_bps,
                benchmark_symbol=benchmark,
                base_currency=base_currency,
                risk_free_rate=risk_free_rate,
                lookback_days=lookback_days,
            )
            backtester = Backtester(
                market_data_store=self.portfolio_manager.market_data_store,
                metrics_calculator=self.metrics_calculator,
                fx_rate_on=fx_rate_on,
                price_currencies=price_currencies,
            )
            result = backtester.run(config, strat_objs)
        except ValueError as e:
            return f"❌ {e}"
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Backtest failed")
            return f"❌ Backtest failed: {e}"

        # Surface universe-resolution warnings (e.g. unresolved currencies).
        result.warnings = list(fx_warnings) + list(result.warnings)

        return self._format(result)

    # -- formatting ---------------------------------------------------------

    def _format(self, result) -> str:
        cfg = result.config
        lines: List[str] = []
        lines.append(f"📈 **Backtest {result.price_start} → {result.price_end}**")
        lines.append(
            f"Universe: {', '.join(cfg.symbols)} | "
            f"Initial: {cfg.initial_capital:,.0f} {cfg.base_currency.value} | "
            f"Rebalance: {cfg.rebalance_frequency} | "
            f"Costs: {cfg.transaction_cost_bps:.0f} bps"
        )
        for w in result.warnings:
            lines.append(f"⚠️  {w}")
        lines.append("")

        header = (
            f"{'Strategy':<22}{'Final':>14}{'Total Ret':>11}"
            f"{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>8}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        def pct(x):
            return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "n/a"

        def num(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "n/a"

        for s in result.strategies:
            m = s.metrics or {}
            lines.append(
                f"{s.name[:21]:<22}"
                f"{s.final_value:>14,.0f}"
                f"{pct(m.get('total_return')):>11}"
                f"{pct(m.get('annualized_return')):>9}"
                f"{pct(m.get('volatility')):>8}"
                f"{num(m.get('sharpe_ratio')):>8}"
                f"{pct(m.get('max_drawdown')):>8}"
            )

        if result.benchmark_curve is not None and not result.benchmark_curve.empty:
            bench_final = float(result.benchmark_curve["total_value"].iloc[-1])
            bench_ret = bench_final / cfg.initial_capital - 1
            lines.append("-" * len(header))
            lines.append(
                f"{('Benchmark ' + cfg.benchmark_symbol)[:21]:<22}"
                f"{bench_final:>14,.0f}{pct(bench_ret):>11}"
            )

        best = max(
            result.strategies,
            key=lambda s: (s.metrics or {}).get("total_return", float("-inf")),
        )
        lines.append("")
        lines.append(
            f"🏆 Best total return: **{best.name}** ({pct((best.metrics or {}).get('total_return'))})"
        )
        return "\n".join(lines)
