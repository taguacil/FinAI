"""
Portfolio optimization module using HRP and Markowitz methods.

Provides weight suggestions for portfolio rebalancing with support for
locking specific positions that shouldn't change.

Uses PyPortfolioOpt for optimization algorithms.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pypfopt import HRPOpt, expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from ..data_providers.manager import DataProviderManager
from .models import Currency, Position, PortfolioSnapshot
from .storage import FileBasedStorage

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService


class OptimizationMethod(str, Enum):
    """Available optimization methods."""

    HRP = "hrp"
    MARKOWITZ = "markowitz"


class OptimizationObjective(str, Enum):
    """Optimization objective for Markowitz."""

    MAX_SHARPE = "max_sharpe"  # Maximize Sharpe ratio
    MIN_VOLATILITY = "min_volatility"  # Minimize volatility
    EFFICIENT_RISK = "efficient_risk"  # Target specific volatility


@dataclass
class AssetMetrics:
    """Metrics for an individual asset."""

    symbol: str
    expected_return: float  # Annualized
    volatility: float  # Annualized
    current_weight: float


@dataclass
class RebalancingTrade:
    """Represents a single trade needed to rebalance."""

    symbol: str
    action: str  # "BUY" or "SELL"
    shares: float
    estimated_value: float  # Value in base currency
    estimated_value_native: float  # Value in asset's native currency
    current_weight: float
    target_weight: float
    currency: str  # Asset's native currency


@dataclass
class OptimizationResult:
    """Results from portfolio optimization."""

    method: OptimizationMethod
    weights: Dict[str, float]
    current_weights: Dict[str, float]
    expected_annual_return: Optional[float]
    annual_volatility: float
    sharpe_ratio: Optional[float]
    rebalancing_trades: List[RebalancingTrade] = field(default_factory=list)
    locked_symbols: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    asset_metrics: List[AssetMetrics] = field(default_factory=list)
    cash_weight: float = 0.0  # Weight of cash in portfolio
    target_volatility: Optional[float] = None  # If risk-constrained optimization


class PortfolioOptimizer:
    """
    Optimizes portfolio weights using HRP or Markowitz.

    Supports locking positions that shouldn't change during optimization.
    Handles multi-currency portfolios with FX conversion.
    Uses local snapshot data by default for efficiency.
    """

    def __init__(
        self,
        data_provider: Union[DataProviderManager, "MarketDataService"],
        base_currency: Currency = Currency.USD,
        storage: Optional[FileBasedStorage] = None,
        portfolio_id: Optional[str] = None,
    ):
        # Handle both DataProviderManager and MarketDataService
        if hasattr(data_provider, "data_manager"):
            self.data_provider = data_provider.data_manager
            self.market_data_service = data_provider
        else:
            self.data_provider = data_provider
            self.market_data_service = None
        self.base_currency = base_currency
        self.storage = storage
        self.portfolio_id = portfolio_id
        self.logger = logging.getLogger(__name__)

    def optimize(
        self,
        positions: Dict[str, Position],
        locked_symbols: Optional[List[str]] = None,
        method: OptimizationMethod = OptimizationMethod.HRP,
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
        total_portfolio_value: Optional[Decimal] = None,
        cash_balances: Optional[Dict[Currency, Decimal]] = None,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        target_volatility: Optional[float] = None,
        include_cash: bool = True,
        optimization_currency: Optional[Currency] = None,
        return_adjustment: float = 0.0,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            positions: Current portfolio positions (symbol -> Position)
            locked_symbols: Symbols to keep at current weights
            method: Optimization method (HRP or Markowitz)
            lookback_days: Days of historical data for covariance
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            total_portfolio_value: Total portfolio value for trade calculations
            cash_balances: Cash balances by currency (included in total value)
            objective: Optimization objective (max_sharpe, min_volatility, efficient_risk)
            target_volatility: Target volatility for efficient_risk objective
            include_cash: Whether to include cash in the portfolio. When True, enables
                          cash allocation to achieve target volatility and includes
                          existing cash balances in rebalancing calculations.
            optimization_currency: Currency to use for optimization (converts all prices).
                                   If None, uses base_currency. Different currencies will
                                   show different volatility due to FX effects.
            return_adjustment: Adjustment to expected returns for scenario analysis
                               (e.g., 0.05 adds 5% to all expected returns).

        Returns:
            OptimizationResult with target weights and rebalancing trades
        """
        locked_symbols = locked_symbols or []
        cash_balances = cash_balances or {}
        warnings: List[str] = []

        # Filter out positions with zero quantity
        active_positions = {
            sym: pos for sym, pos in positions.items() if pos.quantity > 0
        }

        if len(active_positions) < 2:
            raise ValueError("Need at least 2 positions to optimize")

        # Calculate cash value in base currency
        cash_value_base = self._convert_cash_to_base(cash_balances)

        # Calculate current weights (including cash if present)
        current_weights, cash_weight = self._calculate_current_weights_with_cash(
            active_positions, cash_value_base
        )

        # Separate locked and unlocked positions
        locked_weight_sum = sum(
            current_weights.get(sym, 0) for sym in locked_symbols if sym in current_weights
        )
        unlocked_symbols = [sym for sym in active_positions.keys() if sym not in locked_symbols]

        if not unlocked_symbols:
            raise ValueError("All positions are locked - nothing to optimize")

        if len(unlocked_symbols) < 2:
            warnings.append(
                f"Only {len(unlocked_symbols)} unlocked position(s) - limited optimization possible"
            )

        # Determine optimization currency
        opt_currency = optimization_currency or self.base_currency
        if opt_currency != self.base_currency:
            self.logger.info(f"Optimizing in {opt_currency.value} (portfolio base: {self.base_currency.value})")
            warnings.append(
                f"Optimizing in {opt_currency.value}. Returns and volatility are measured "
                f"from a {opt_currency.value} investor's perspective."
            )

        # Fetch historical prices
        prices_df = self._fetch_prices(list(active_positions.keys()), lookback_days)

        if prices_df.empty:
            raise ValueError("Could not fetch historical price data")

        # Convert prices to optimization currency if needed
        if opt_currency != self.base_currency:
            prices_df = self._convert_prices_to_currency(
                prices_df, active_positions, opt_currency
            )
            if prices_df.empty:
                raise ValueError(
                    f"Could not convert prices to {opt_currency.value}. "
                    "FX rate data may be unavailable."
                )

        # Check for missing symbols
        available_symbols = set(prices_df.columns)
        missing = set(active_positions.keys()) - available_symbols
        if missing:
            warnings.append(f"No price data for: {', '.join(missing)}")
            # Remove missing symbols from optimization
            unlocked_symbols = [s for s in unlocked_symbols if s in available_symbols]
            if not unlocked_symbols:
                raise ValueError("No price data available for unlocked positions")

        # Calculate individual asset metrics for plotting
        asset_metrics = self._calculate_asset_metrics(prices_df, current_weights)

        # Run optimization on unlocked portion
        # The optimizer returns weights that sum to 1.0 for the unlocked/risky portion
        unlocked_prices = prices_df[unlocked_symbols]

        self.logger.info(f"Running {method.value} optimization on {len(unlocked_symbols)} assets")
        self.logger.info(f"Price data: {len(unlocked_prices)} days, columns: {list(unlocked_prices.columns)}")

        if method == OptimizationMethod.HRP:
            raw_weights = self._run_hrp(unlocked_prices)
        else:
            raw_weights = self._run_markowitz_standard(
                unlocked_prices, risk_free_rate, objective, return_adjustment
            )

        self.logger.info(f"Raw weights from optimizer (sum={sum(raw_weights.values()):.4f}): {raw_weights}")

        # Calculate metrics for the pure risky portfolio (weights sum to 1.0)
        # This is before any cash blending or locked position adjustments
        risky_return, risky_volatility, risky_sharpe = self._calculate_metrics(
            unlocked_prices, raw_weights, risk_free_rate
        )

        self.logger.info(
            f"Risky portfolio metrics: Return={risky_return:.2%}, Vol={risky_volatility:.2%}, Sharpe={risky_sharpe:.3f}"
            if risky_return and risky_volatility else "Could not calculate risky portfolio metrics"
        )

        # Step 1: Build full risky portfolio weights (locked + optimized)
        # No cash blending yet - first we need to calculate full portfolio volatility
        remaining_for_risky = 1.0 - locked_weight_sum

        # Build initial target weights (assuming no cash)
        target_weights = {}

        # Add locked positions at their current weights
        for sym in locked_symbols:
            if sym in current_weights:
                target_weights[sym] = current_weights[sym]

        # Add optimized weights scaled to fill remaining space
        for sym, weight in raw_weights.items():
            target_weights[sym] = weight * remaining_for_risky

        # Step 2: Calculate full risky portfolio volatility
        # This includes both locked and optimized positions
        full_risky_weight = sum(target_weights.values())
        full_risky_volatility = None

        if full_risky_weight > 0:
            normalized_full = {sym: w / full_risky_weight for sym, w in target_weights.items()}
            _, full_risky_volatility, _ = self._calculate_metrics(
                prices_df, normalized_full, risk_free_rate
            )
            self.logger.info(f"Full risky portfolio volatility: {full_risky_volatility:.2%}" if full_risky_volatility else "")

        # Step 3: Determine cash allocation based on target volatility
        risky_fraction = 1.0
        target_cash_total = 0.0

        if target_volatility is not None:
            if full_risky_volatility is not None and full_risky_volatility > 0:
                if target_volatility < full_risky_volatility:
                    if not include_cash:
                        # Cannot achieve target volatility without cash
                        raise ValueError(
                            f"Target volatility {target_volatility:.1%} is below the minimum "
                            f"achievable with risky assets only ({full_risky_volatility:.1%}). "
                            f"Enable 'Include cash in rebalancing' to achieve lower volatility."
                        )
                    # Need to blend with cash to reduce volatility
                    # Portfolio vol = risky_fraction * full_risky_vol
                    # Solve: target_vol = risky_fraction * full_risky_vol
                    risky_fraction = target_volatility / full_risky_volatility
                    risky_fraction = max(0.0, min(1.0, risky_fraction))

                    warnings.append(
                        f"To achieve {target_volatility:.1%} volatility, "
                        f"scaling portfolio to {risky_fraction:.1%} risky assets. "
                        f"(Full risky portfolio: {full_risky_volatility:.1%} vol)"
                    )
                else:
                    # Target volatility is higher than what's achievable
                    warnings.append(
                        f"Target volatility {target_volatility:.1%} is higher than the "
                        f"optimized portfolio's {full_risky_volatility:.1%}. "
                        f"Keeping full allocation to risky assets (no cash needed)."
                    )

        # Step 4: Scale ALL positions (including locked) by risky_fraction
        # This ensures consistent volatility reduction across the portfolio
        if risky_fraction < 1.0:
            target_weights = {sym: w * risky_fraction for sym, w in target_weights.items()}
            target_cash_total = 1.0 - sum(target_weights.values())

            if locked_symbols:
                warnings.append(
                    "Note: Locked positions have been scaled down to achieve target volatility. "
                    "To keep locked positions at exact current weights, disable target volatility."
                )
        else:
            target_cash_total = 0.0

        allocated_to_risky = sum(w for sym, w in target_weights.items() if sym not in locked_symbols)
        final_locked_weight = sum(w for sym, w in target_weights.items() if sym in locked_symbols)

        self.logger.info(
            f"Weight allocation: locked={final_locked_weight:.2%}, "
            f"optimized={allocated_to_risky:.2%}, cash={target_cash_total:.2%}, "
            f"sum={final_locked_weight + allocated_to_risky + target_cash_total:.4f}"
        )

        # Calculate final portfolio metrics on ALL positions with target weights
        # This correctly accounts for both locked and optimized positions
        all_risky_weight = sum(target_weights.values())

        if all_risky_weight > 0:
            # Normalize weights to sum to 1.0 for metrics calculation
            normalized_weights = {sym: w / all_risky_weight for sym, w in target_weights.items()}

            # Calculate metrics on the normalized risky portfolio
            portfolio_return, portfolio_volatility, _ = self._calculate_metrics(
                prices_df, normalized_weights, risk_free_rate
            )

            if portfolio_return is not None and portfolio_volatility is not None:
                # Scale by actual risky exposure and add cash component
                expected_return = (
                    all_risky_weight * portfolio_return + target_cash_total * risk_free_rate
                )
                volatility = all_risky_weight * portfolio_volatility

                # Sharpe ratio
                if volatility > 0:
                    sharpe = (expected_return - risk_free_rate) / volatility
                else:
                    sharpe = None

                self.logger.info(
                    f"Final portfolio: Return={expected_return:.2%}, Vol={volatility:.2%}, "
                    f"Sharpe={sharpe:.3f}" if sharpe is not None else
                    f"Final portfolio: Return={expected_return:.2%}, Vol={volatility:.2%}, Sharpe=N/A"
                )
            else:
                expected_return, volatility, sharpe = None, 0.0, None
        else:
            expected_return, volatility, sharpe = None, 0.0, None

        # Calculate rebalancing trades with FX conversion
        rebalancing_trades = []
        if total_portfolio_value:
            # Add existing cash to total for weight calculations
            effective_total = total_portfolio_value
            if include_cash:
                effective_total += cash_value_base

            rebalancing_trades = self._calculate_rebalancing_trades(
                active_positions, target_weights, effective_total, current_weights
            )

            # Add cash allocation trade if there's a difference between current and target cash
            if include_cash and effective_total > 0:
                current_cash_weight_in_total = float(cash_value_base / effective_total)
                cash_diff = target_cash_total - current_cash_weight_in_total

                if abs(cash_diff) > 0.01:
                    cash_value_change = float(effective_total) * abs(cash_diff)

                    # Determine action based on direction
                    if cash_diff > 0:
                        action = "HOLD"  # Need to hold more cash
                    else:
                        action = "DEPLOY"  # Deploy existing cash into risky assets

                    rebalancing_trades.append(
                        RebalancingTrade(
                            symbol=f"CASH ({self.base_currency.value})",
                            action=action,
                            shares=0.0,
                            estimated_value=cash_value_change,
                            estimated_value_native=cash_value_change,
                            current_weight=current_cash_weight_in_total,
                            target_weight=target_cash_total,
                            currency=self.base_currency.value,
                        )
                    )

        return OptimizationResult(
            method=method,
            weights=target_weights,
            current_weights=current_weights,
            expected_annual_return=expected_return,
            annual_volatility=volatility,
            sharpe_ratio=sharpe,
            rebalancing_trades=rebalancing_trades,
            locked_symbols=locked_symbols,
            warnings=warnings,
            asset_metrics=asset_metrics,
            cash_weight=target_cash_total,
            target_volatility=target_volatility,
        )

    def compare_methods(
        self,
        positions: Dict[str, Position],
        locked_symbols: Optional[List[str]] = None,
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
        total_portfolio_value: Optional[Decimal] = None,
        cash_balances: Optional[Dict[Currency, Decimal]] = None,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        target_volatility: Optional[float] = None,
        include_cash: bool = True,
        optimization_currency: Optional[Currency] = None,
    ) -> Dict[OptimizationMethod, OptimizationResult]:
        """
        Run both HRP and Markowitz, return comparison.

        Args:
            positions: Current portfolio positions
            locked_symbols: Symbols to keep at current weights
            lookback_days: Days of historical data
            risk_free_rate: Annual risk-free rate
            total_portfolio_value: Total value for trade calculations
            cash_balances: Cash balances by currency
            objective: Optimization objective for Markowitz
            target_volatility: Target volatility for efficient_risk objective
            include_cash: Whether to include cash in portfolio for volatility targeting and rebalancing
            optimization_currency: Currency to use for optimization (converts all prices)

        Returns:
            Dict mapping method to its OptimizationResult
        """
        results = {}

        for method in [OptimizationMethod.HRP, OptimizationMethod.MARKOWITZ]:
            try:
                results[method] = self.optimize(
                    positions=positions,
                    locked_symbols=locked_symbols,
                    method=method,
                    lookback_days=lookback_days,
                    risk_free_rate=risk_free_rate,
                    total_portfolio_value=total_portfolio_value,
                    cash_balances=cash_balances,
                    objective=objective,
                    target_volatility=target_volatility,
                    include_cash=include_cash,
                    optimization_currency=optimization_currency,
                )
            except Exception as e:
                self.logger.warning(f"Failed to run {method.value} optimization: {e}")
                # Create a result with error
                results[method] = OptimizationResult(
                    method=method,
                    weights={},
                    current_weights={},
                    expected_annual_return=None,
                    annual_volatility=0,
                    sharpe_ratio=None,
                    warnings=[f"Optimization failed: {str(e)}"],
                )

        return results

    def _convert_cash_to_base(
        self, cash_balances: Dict[Currency, Decimal]
    ) -> Decimal:
        """Convert all cash balances to base currency."""
        total_base = Decimal("0")

        for currency, amount in cash_balances.items():
            if currency == self.base_currency:
                total_base += amount
            else:
                # Get FX rate
                fx_rate = self._get_fx_rate(currency, self.base_currency)
                total_base += amount * fx_rate

        return total_base

    def _get_fx_rate(
        self, from_currency: Currency, to_currency: Currency
    ) -> Decimal:
        """Get FX rate from local snapshot data only (no network calls).

        Uses the most recent FX rate found in snapshots, or 1.0 if unavailable.
        """
        if from_currency == to_currency:
            return Decimal("1")

        # Try to get FX rate from latest snapshot's position data
        # Positions in foreign currency implicitly contain FX info via their base value
        if self.storage and self.portfolio_id:
            try:
                latest_snap = self.storage.get_latest_snapshot(self.portfolio_id)
                if latest_snap:
                    # Look for a position in the from_currency to estimate rate
                    # by comparing market_value (native) to what we'd expect in base
                    for pos in latest_snap.positions.values():
                        if pos.instrument and pos.instrument.currency == from_currency:
                            if pos.current_price and pos.quantity > 0:
                                native_value = pos.quantity * pos.current_price
                                if pos.market_value and native_value > 0:
                                    # market_value might be in base currency
                                    # This is an approximation
                                    pass
            except Exception as e:
                self.logger.debug(f"Could not derive FX from snapshots: {e}")

        # For optimization purposes, use 1.0 - the weights are relative anyway
        # FX only matters for absolute trade values, not weight calculations
        self.logger.debug(
            f"Using 1.0 for {from_currency.value}/{to_currency.value} (local only mode)"
        )
        return Decimal("1")

    def _get_position_value_in_base(self, position: Position) -> Decimal:
        """Get position value converted to base currency."""
        if position.current_price and position.quantity > 0:
            value = position.quantity * position.current_price
        elif position.average_cost and position.quantity > 0:
            value = position.quantity * position.average_cost
        else:
            return Decimal("0")

        # Convert to base currency
        pos_currency = position.instrument.currency if position.instrument else self.base_currency
        if pos_currency != self.base_currency:
            fx_rate = self._get_fx_rate(pos_currency, self.base_currency)
            value = value * fx_rate

        return value

    def _calculate_current_weights_with_cash(
        self, positions: Dict[str, Position], cash_value_base: Decimal
    ) -> Tuple[Dict[str, float], float]:
        """Calculate current portfolio weights including cash."""
        total_value = cash_value_base
        position_values: Dict[str, Decimal] = {}

        for symbol, pos in positions.items():
            value = self._get_position_value_in_base(pos)
            position_values[symbol] = value
            total_value += value

        if total_value == 0:
            return {sym: 0.0 for sym in positions.keys()}, 0.0

        weights = {
            sym: float(val / total_value) for sym, val in position_values.items()
        }
        cash_weight = float(cash_value_base / total_value) if cash_value_base > 0 else 0.0

        return weights, cash_weight

    def _calculate_asset_metrics(
        self, prices_df: pd.DataFrame, current_weights: Dict[str, float]
    ) -> List[AssetMetrics]:
        """Calculate individual asset metrics for the volatility-return plot."""
        asset_metrics = []

        returns = prices_df.pct_change().dropna()

        for symbol in prices_df.columns:
            if symbol not in returns.columns:
                continue

            # Annualized metrics (252 trading days)
            expected_return = float(returns[symbol].mean() * 252)
            volatility = float(returns[symbol].std() * np.sqrt(252))
            current_weight = current_weights.get(symbol, 0.0)

            asset_metrics.append(
                AssetMetrics(
                    symbol=symbol,
                    expected_return=expected_return,
                    volatility=volatility,
                    current_weight=current_weight,
                )
            )

        return asset_metrics

    def _calculate_current_weights(
        self, positions: Dict[str, Position]
    ) -> Dict[str, float]:
        """Calculate current portfolio weights based on market values (no cash)."""
        weights, _ = self._calculate_current_weights_with_cash(positions, Decimal("0"))
        return weights

    def _fetch_prices(
        self, symbols: List[str], lookback_days: int
    ) -> pd.DataFrame:
        """Fetch historical prices from local snapshots only (no network calls)."""
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer

        if not self.storage or not self.portfolio_id:
            raise ValueError(
                "Optimizer requires storage and portfolio_id to fetch local price data. "
                "Please ensure snapshots are available for this portfolio."
            )

        prices_df = self._fetch_prices_from_snapshots(symbols, start_date, end_date)

        if prices_df.empty:
            raise ValueError(
                f"No local snapshot data found for period {start_date} to {end_date}. "
                "Please create snapshots using 'Update Snapshots' in the Portfolio tab first."
            )

        self.logger.info(
            f"Using local snapshot data: {len(prices_df)} days for {len(prices_df.columns)} symbols"
        )

        if len(prices_df) < 30:
            self.logger.warning(
                f"Only {len(prices_df)} days of local data available. "
                "Consider creating more snapshots for better optimization results."
            )

        return prices_df

    def _fetch_prices_from_snapshots(
        self, symbols: List[str], start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Extract historical prices from local portfolio snapshots."""
        try:
            snapshots = self.storage.load_snapshots(
                self.portfolio_id, start_date, end_date
            )

            if not snapshots:
                return pd.DataFrame()

            # Build price series from snapshots
            prices_dict: Dict[str, Dict[date, float]] = {sym: {} for sym in symbols}

            for snap in snapshots:
                snap_date = snap.date
                for symbol in symbols:
                    if symbol in snap.positions:
                        pos = snap.positions[symbol]
                        if pos.current_price is not None:
                            prices_dict[symbol][snap_date] = float(pos.current_price)

            # Convert to DataFrame
            if not any(prices_dict.values()):
                return pd.DataFrame()

            # Create series for each symbol
            series_dict = {}
            for symbol, date_prices in prices_dict.items():
                if date_prices:
                    series_dict[symbol] = pd.Series(date_prices)

            if not series_dict:
                return pd.DataFrame()

            prices_df = pd.DataFrame(series_dict)
            prices_df = prices_df.sort_index()

            # Forward fill missing values
            prices_df = prices_df.ffill().dropna()

            return prices_df

        except Exception as e:
            self.logger.warning(f"Failed to load prices from snapshots: {e}")
            return pd.DataFrame()

    def _convert_prices_to_currency(
        self,
        prices_df: pd.DataFrame,
        positions: Dict[str, Position],
        target_currency: Currency,
    ) -> pd.DataFrame:
        """Convert price series to target currency using historical FX rates.

        For each asset, determines its native currency and applies the appropriate
        FX conversion. This allows optimization from different currency perspectives.

        Args:
            prices_df: DataFrame with prices in native currencies (symbol columns, date index)
            positions: Position dict to get each asset's native currency
            target_currency: Currency to convert all prices to

        Returns:
            DataFrame with all prices converted to target_currency
        """
        if prices_df.empty:
            return prices_df

        converted_prices = {}
        dates = prices_df.index.tolist()

        for symbol in prices_df.columns:
            pos = positions.get(symbol)
            if not pos:
                continue

            # Get asset's native currency
            asset_currency = self.base_currency
            if pos.instrument and pos.instrument.currency:
                asset_currency = pos.instrument.currency

            if asset_currency == target_currency:
                # No conversion needed
                converted_prices[symbol] = prices_df[symbol]
                continue

            # Convert each price using historical FX rate
            converted_series = {}
            fx_cache: Dict[date, Decimal] = {}

            for dt in dates:
                price = prices_df.loc[dt, symbol]
                if pd.isna(price):
                    continue

                # Get FX rate for this date (with caching)
                if dt not in fx_cache:
                    fx_rate = self._get_historical_fx_rate(dt, asset_currency, target_currency)
                    fx_cache[dt] = fx_rate

                fx_rate = fx_cache[dt]
                if fx_rate and fx_rate > 0:
                    converted_series[dt] = float(price) * float(fx_rate)

            if converted_series:
                converted_prices[symbol] = pd.Series(converted_series)
            else:
                self.logger.warning(f"Could not convert {symbol} prices to {target_currency.value}")

        if not converted_prices:
            return pd.DataFrame()

        result_df = pd.DataFrame(converted_prices)
        result_df = result_df.sort_index().ffill().dropna()

        self.logger.info(
            f"Converted {len(result_df.columns)} assets to {target_currency.value} "
            f"({len(result_df)} days)"
        )

        return result_df

    def _get_historical_fx_rate(
        self, dt: date, from_currency: Currency, to_currency: Currency
    ) -> Optional[Decimal]:
        """Get historical FX rate for a specific date.

        Uses the data provider's historical FX method with fallback to current rate.
        """
        if from_currency == to_currency:
            return Decimal("1")

        try:
            # Try to get historical rate from data provider
            rate = self.data_provider.get_historical_fx_rate_on(dt, from_currency, to_currency)
            if rate and rate > 0:
                return rate
        except Exception as e:
            self.logger.debug(f"Could not get historical FX for {dt}: {e}")

        # Fallback: try current rate (less accurate but better than nothing)
        try:
            rate = self.data_provider.get_exchange_rate(from_currency, to_currency)
            if rate and rate > 0:
                return rate
        except Exception as e:
            self.logger.debug(f"Could not get current FX rate: {e}")

        return None

    def _run_hrp(self, prices: pd.DataFrame) -> Dict[str, float]:
        """Run Hierarchical Risk Parity optimization using pypfopt."""
        # Calculate returns for HRP
        returns = prices.pct_change().dropna()

        hrp = HRPOpt(returns)
        weights = hrp.optimize()

        return dict(weights)

    def _run_markowitz_standard(
        self,
        prices: pd.DataFrame,
        risk_free_rate: float,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        return_adjustment: float = 0.0,
    ) -> Dict[str, float]:
        """Run standard Markowitz optimization.

        For EFFICIENT_RISK objective, we use MAX_SHARPE since cash blending
        happens after optimization to achieve target volatility.

        Args:
            prices: Historical price data
            risk_free_rate: Annual risk-free rate
            objective: Optimization objective
            return_adjustment: Adjustment to add to all expected returns (for scenario analysis)

        Returns:
            Dict of symbol -> weight (sums to 1.0)
        """
        mu = expected_returns.mean_historical_return(prices)

        # Apply return adjustment for scenario analysis
        if return_adjustment != 0.0:
            mu = mu + return_adjustment

        cov = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, cov)

        try:
            if objective == OptimizationObjective.MIN_VOLATILITY:
                ef.min_volatility()
            else:
                # For MAX_SHARPE and EFFICIENT_RISK, use max_sharpe
                # EFFICIENT_RISK: we get max Sharpe portfolio, then blend with cash
                # This is correct because the capital allocation line from
                # risk-free to max-Sharpe portfolio dominates all other portfolios
                ef.max_sharpe(risk_free_rate=risk_free_rate)

        except Exception as e:
            self.logger.warning(f"Optimization failed ({objective.value}): {e}. Trying min_volatility.")
            ef = EfficientFrontier(mu, cov)
            try:
                ef.min_volatility()
            except Exception as e2:
                self.logger.error(f"All optimization attempts failed: {e2}. Using equal weights.")
                n = len(prices.columns)
                return {sym: 1.0 / n for sym in prices.columns}

        weights = ef.clean_weights()
        return dict(weights)

    def _run_markowitz(
        self,
        prices: pd.DataFrame,
        risk_free_rate: float,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        target_volatility: Optional[float] = None,
    ) -> Dict[str, float]:
        """Run mean-variance optimization using pypfopt.

        Args:
            prices: Historical price data
            risk_free_rate: Annual risk-free rate
            objective: Optimization objective
            target_volatility: Target annual volatility (for EFFICIENT_RISK)

        Returns:
            Dict of symbol -> weight (may include "_CASH" for cash allocation)
        """
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(prices)
        cov = risk_models.sample_cov(prices)

        # For EFFICIENT_RISK with low target volatility, we may need to include cash
        if objective == OptimizationObjective.EFFICIENT_RISK and target_volatility is not None:
            # Calculate minimum achievable volatility with current assets
            try:
                ef_test = EfficientFrontier(mu, cov)
                ef_test.min_volatility()
                min_vol_weights = ef_test.clean_weights()

                # Calculate the minimum volatility
                weight_arr = np.array([min_vol_weights.get(s, 0) for s in prices.columns])
                cov_matrix = cov.values
                min_vol = float(np.sqrt(np.dot(weight_arr.T, np.dot(cov_matrix, weight_arr))))

                if target_volatility < min_vol:
                    # Target is below minimum - need to include cash
                    self.logger.info(
                        f"Target volatility {target_volatility:.1%} is below minimum {min_vol:.1%}. "
                        "Including cash allocation to achieve target."
                    )
                    return self._run_markowitz_with_cash(
                        prices, risk_free_rate, target_volatility, min_vol
                    )
            except Exception as e:
                self.logger.warning(f"Could not calculate min volatility: {e}")

        # Standard optimization without cash
        ef = EfficientFrontier(mu, cov)

        try:
            if objective == OptimizationObjective.MIN_VOLATILITY:
                ef.min_volatility()

            elif objective == OptimizationObjective.EFFICIENT_RISK:
                if target_volatility is None:
                    target_volatility = 0.15
                    self.logger.info(f"No target volatility specified, using {target_volatility:.1%}")

                try:
                    ef.efficient_risk(target_volatility)
                except ValueError as e:
                    self.logger.warning(
                        f"Target volatility {target_volatility:.1%} not feasible: {e}. "
                        "Trying with cash allocation."
                    )
                    # Try with cash
                    ef_min = EfficientFrontier(mu, cov)
                    ef_min.min_volatility()
                    min_vol_weights = ef_min.clean_weights()
                    weight_arr = np.array([min_vol_weights.get(s, 0) for s in prices.columns])
                    cov_matrix = cov.values
                    min_vol = float(np.sqrt(np.dot(weight_arr.T, np.dot(cov_matrix, weight_arr))))

                    return self._run_markowitz_with_cash(
                        prices, risk_free_rate, target_volatility, min_vol
                    )

            else:  # MAX_SHARPE (default)
                ef.max_sharpe(risk_free_rate=risk_free_rate)

        except Exception as e:
            self.logger.warning(f"Optimization failed ({objective.value}): {e}. Trying min_volatility.")
            ef = EfficientFrontier(mu, cov)
            try:
                ef.min_volatility()
            except Exception as e2:
                self.logger.error(f"All optimization attempts failed: {e2}. Using equal weights.")
                n = len(prices.columns)
                return {sym: 1.0 / n for sym in prices.columns}

        weights = ef.clean_weights()
        return dict(weights)

    def _run_markowitz_with_cash(
        self,
        prices: pd.DataFrame,
        risk_free_rate: float,
        target_volatility: float,
        min_risky_volatility: float,
    ) -> Dict[str, float]:
        """Run Markowitz optimization including cash to achieve lower volatility.

        Uses the two-fund separation theorem: combine the min-volatility risky
        portfolio with cash to achieve any volatility below the minimum risky volatility.

        Args:
            prices: Historical price data
            risk_free_rate: Annual risk-free rate
            target_volatility: Target annual volatility
            min_risky_volatility: Minimum volatility achievable with risky assets only

        Returns:
            Dict of symbol -> weight, including "_CASH" for cash allocation
        """
        # Get the minimum volatility portfolio of risky assets
        mu = expected_returns.mean_historical_return(prices)
        cov = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, cov)
        ef.min_volatility()
        risky_weights = ef.clean_weights()

        # Calculate the fraction to invest in risky assets
        # Portfolio vol = risky_fraction * risky_vol (cash has 0 vol)
        # target_vol = risky_fraction * min_risky_vol
        # risky_fraction = target_vol / min_risky_vol

        if min_risky_volatility <= 0:
            # Edge case: risky assets have no volatility (unlikely)
            risky_fraction = 1.0
        else:
            risky_fraction = target_volatility / min_risky_volatility
            risky_fraction = min(1.0, max(0.0, risky_fraction))  # Clamp to [0, 1]

        cash_fraction = 1.0 - risky_fraction

        self.logger.info(
            f"To achieve {target_volatility:.1%} volatility: "
            f"{risky_fraction:.1%} in risky assets, {cash_fraction:.1%} in cash"
        )

        # Scale risky weights by risky_fraction
        result_weights = {}
        for symbol, weight in risky_weights.items():
            scaled_weight = weight * risky_fraction
            if scaled_weight > 0.0001:  # Only include meaningful weights
                result_weights[symbol] = scaled_weight

        # Add cash allocation
        if cash_fraction > 0.0001:
            result_weights["_CASH"] = cash_fraction

        return result_weights

    def _calculate_metrics(
        self,
        prices: pd.DataFrame,
        weights: Dict[str, float],
        risk_free_rate: float,
    ) -> Tuple[Optional[float], float, Optional[float]]:
        """Calculate expected return, volatility, and Sharpe ratio for given weights."""
        # Filter to available symbols
        available = [s for s in weights.keys() if s in prices.columns]
        if not available:
            return None, 0.0, None

        prices_subset = prices[available]

        # Calculate returns
        returns = prices_subset.pct_change().dropna()

        weight_arr = np.array([weights[s] for s in available])

        # Annualized expected return (252 trading days)
        mean_returns = returns.mean() * 252
        expected_return = float(np.dot(weight_arr, mean_returns))

        # Annualized volatility
        cov_matrix = returns.cov() * 252
        portfolio_var = np.dot(weight_arr.T, np.dot(cov_matrix, weight_arr))
        volatility = float(np.sqrt(portfolio_var))

        # Sharpe ratio
        if volatility > 0:
            sharpe = (expected_return - risk_free_rate) / volatility
        else:
            sharpe = None

        return expected_return, volatility, sharpe

    def _calculate_metrics_with_cash(
        self,
        prices: pd.DataFrame,
        weights: Dict[str, float],
        risk_free_rate: float,
        cash_weight: float,
    ) -> Tuple[Optional[float], float, Optional[float]]:
        """Calculate metrics accounting for cash allocation.

        Cash has 0 volatility and earns the risk-free rate.
        Portfolio metrics are blended between risky assets and cash.
        """
        # Get metrics for risky portion only
        risky_return, risky_volatility, _ = self._calculate_metrics(
            prices, weights, risk_free_rate
        )

        if risky_return is None:
            return None, 0.0, None

        # Calculate total weight in risky assets
        risky_weight = sum(weights.values())

        # If no cash, return risky metrics directly
        if cash_weight <= 0 or risky_weight >= 1.0:
            sharpe = None
            if risky_volatility > 0:
                sharpe = (risky_return - risk_free_rate) / risky_volatility
            return risky_return, risky_volatility, sharpe

        # Blend returns: risky portion earns risky_return, cash earns risk_free_rate
        # Total portfolio return = risky_weight * risky_return + cash_weight * risk_free_rate
        total_weight = risky_weight + cash_weight
        blended_return = (risky_weight * risky_return + cash_weight * risk_free_rate) / total_weight

        # Blend volatility: cash has 0 volatility, uncorrelated with risky assets
        # Portfolio vol = risky_weight * risky_vol (simplified - cash adds no vol)
        blended_volatility = (risky_weight / total_weight) * risky_volatility

        # Sharpe ratio with blended metrics
        if blended_volatility > 0:
            sharpe = (blended_return - risk_free_rate) / blended_volatility
        else:
            sharpe = None

        self.logger.info(
            f"Portfolio with {cash_weight:.1%} cash: "
            f"Return {blended_return:.2%}, Vol {blended_volatility:.2%}"
        )

        return blended_return, blended_volatility, sharpe

    def _calculate_rebalancing_trades(
        self,
        positions: Dict[str, Position],
        target_weights: Dict[str, float],
        total_value: Decimal,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> List[RebalancingTrade]:
        """Calculate trades needed to reach target allocation with FX conversion.

        Args:
            positions: Current portfolio positions
            target_weights: Target weights for each symbol
            total_value: Total portfolio value (including cash if applicable)
            current_weights: Pre-calculated current weights (including cash in denominator).
                           If None, calculates from positions only (legacy behavior).
        """
        trades = []
        if current_weights is None:
            current_weights = self._calculate_current_weights(positions)

        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) < 0.001:  # Skip tiny differences
                continue

            # Calculate value change in base currency
            value_change_base = float(total_value) * weight_diff

            # Get position info for currency and price
            pos = positions.get(symbol)
            pos_currency = self.base_currency
            if pos and pos.instrument:
                pos_currency = pos.instrument.currency

            # Convert value change to native currency for share calculation
            if pos_currency != self.base_currency:
                fx_rate = self._get_fx_rate(self.base_currency, pos_currency)
                value_change_native = value_change_base * float(fx_rate)
            else:
                value_change_native = value_change_base

            # Get current price for share calculation
            if pos and pos.current_price:
                price = float(pos.current_price)
                shares = abs(value_change_native / price)
            else:
                shares = 0.0

            action = "BUY" if weight_diff > 0 else "SELL"

            trades.append(
                RebalancingTrade(
                    symbol=symbol,
                    action=action,
                    shares=round(shares, 2),
                    estimated_value=abs(value_change_base),  # Base currency
                    estimated_value_native=abs(value_change_native),  # Native currency
                    current_weight=current_weight,
                    target_weight=target_weight,
                    currency=pos_currency.value if pos_currency else self.base_currency.value,
                )
            )

        # Sort by absolute value change (largest trades first)
        trades.sort(key=lambda t: t.estimated_value, reverse=True)

        return trades
