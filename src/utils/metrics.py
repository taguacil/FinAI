"""
Financial metrics calculator for portfolio analysis.
"""

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data_providers.manager import DataProviderManager
from ..portfolio.models import PortfolioSnapshot


class FinancialMetricsCalculator:
    """Calculator for various financial metrics and portfolio analysis."""

    def __init__(self, data_manager: Optional[DataProviderManager] = None):
        """Initialize metrics calculator."""
        self.data_manager = data_manager or DataProviderManager()

    def calculate_returns(self, snapshots: List[PortfolioSnapshot], cash_flows_by_day: Optional[Dict[date, float]] = None) -> List[float]:
        """Calculate daily time-weighted returns (ignoring external cash injections/withdrawals).

        If cash_flows_by_day is provided (base currency amounts, positive for deposits,
        negative for withdrawals), the return for day t is computed as:
        r_t = (V_t - V_{t-1} - CF_t) / V_{t-1}, where CF_t is external cash flow on day t.
        """
        if len(snapshots) < 2:
            return []

        returns = []
        for i in range(1, len(snapshots)):
            prev_value = float(snapshots[i - 1].total_value)
            curr_value = float(snapshots[i].total_value)
            cf = 0.0
            if cash_flows_by_day:
                cf = float(cash_flows_by_day.get(snapshots[i].date, 0.0))

            if prev_value > 0:
                daily_return = (curr_value - prev_value - cf) / prev_value
                returns.append(daily_return)

        return returns

    def calculate_volatility(
        self, returns: List[float], annualized: bool = True
    ) -> float:
        """Calculate portfolio volatility."""
        # Filter out exact-zero returns (e.g., weekends/holidays due to forward-filled prices)
        filtered = [r for r in returns if abs(r) > 1e-12]
        if len(filtered) < 2:
            return 0.0

        volatility = np.std(filtered, ddof=1)

        if annualized:
            # Annualize assuming 252 trading days
            volatility *= np.sqrt(252)

        return float(volatility)

    def calculate_sharpe_ratio(
        self, returns: List[float], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)

        if volatility == 0:
            return 0.0

        # Annualize
        annualized_return = avg_return * 252
        annualized_volatility = volatility * np.sqrt(252)

        sharpe = (annualized_return - risk_free_rate) / annualized_volatility
        return float(sharpe)

    def calculate_max_drawdown(
        self, snapshots: List[PortfolioSnapshot]
    ) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if len(snapshots) < 2:
            return 0.0, 0

        values = [float(s.total_value) for s in snapshots]
        peak = values[0]
        max_drawdown = 0.0
        max_duration = 0
        current_duration = 0

        for value in values:
            if value > peak:
                peak = value
                current_duration = 0
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
                current_duration += 1
                max_duration = max(max_duration, current_duration)

        return max_drawdown, max_duration

    def calculate_beta(
        self, portfolio_returns: List[float], benchmark_returns: List[float]
    ) -> float:
        """Calculate beta against a benchmark."""
        if (
            len(portfolio_returns) != len(benchmark_returns)
            or len(portfolio_returns) < 2
        ):
            return 0.0

        # Calculate covariance and variance
        portfolio_array = np.array(portfolio_returns)
        benchmark_array = np.array(benchmark_returns)

        covariance = np.cov(portfolio_array, benchmark_array)[0, 1]
        benchmark_variance = np.var(benchmark_array, ddof=1)

        if benchmark_variance == 0:
            return 0.0

        beta = covariance / benchmark_variance
        return float(beta)

    def calculate_alpha(
        self,
        portfolio_returns: List[float],
        benchmark_returns: List[float],
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate Jensen's alpha."""
        if (
            len(portfolio_returns) != len(benchmark_returns)
            or len(portfolio_returns) < 2
        ):
            return 0.0

        beta = self.calculate_beta(portfolio_returns, benchmark_returns)

        avg_portfolio_return = np.mean(portfolio_returns) * 252  # Annualized
        avg_benchmark_return = np.mean(benchmark_returns) * 252  # Annualized

        alpha = avg_portfolio_return - (
            risk_free_rate + beta * (avg_benchmark_return - risk_free_rate)
        )
        return float(alpha)

    def calculate_information_ratio(
        self, portfolio_returns: List[float], benchmark_returns: List[float]
    ) -> float:
        """Calculate information ratio (active return / tracking error)."""
        if (
            len(portfolio_returns) != len(benchmark_returns)
            or len(portfolio_returns) < 2
        ):
            return 0.0

        # Calculate excess returns
        excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)

        avg_excess_return = np.mean(excess_returns)
        tracking_error = np.std(excess_returns, ddof=1)

        if tracking_error == 0:
            return 0.0

        # Annualize
        annualized_excess_return = avg_excess_return * 252
        annualized_tracking_error = tracking_error * np.sqrt(252)

        return float(annualized_excess_return / annualized_tracking_error)

    def calculate_sortino_ratio(
        self,
        returns: List[float],
        target_return: float = 0.0,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)."""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (target_return / 252)  # Daily target

        # Calculate downside deviation (only negative excess returns)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf")  # No downside risk

        downside_deviation = np.sqrt(np.mean(downside_returns**2))

        if downside_deviation == 0:
            return 0.0

        avg_return = np.mean(returns) * 252  # Annualized
        annualized_downside_dev = downside_deviation * np.sqrt(252)

        sortino = (avg_return - risk_free_rate) / annualized_downside_dev
        return float(sortino)

    def calculate_calmar_ratio(
        self, returns: List[float], snapshots: List[PortfolioSnapshot]
    ) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) < 2 or len(snapshots) < 2:
            return 0.0

        annual_return = np.mean(returns) * 252
        max_drawdown, _ = self.calculate_max_drawdown(snapshots)

        if max_drawdown == 0:
            return float("inf")

        return float(annual_return / max_drawdown)

    def get_benchmark_returns(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[float]:
        """Get benchmark returns for comparison."""
        try:
            price_data = self.data_manager.get_historical_prices(
                symbol, start_date, end_date
            )

            if len(price_data) < 2:
                return []

            returns = []
            for i in range(1, len(price_data)):
                prev_price = float(price_data[i - 1].close_price or 0)
                curr_price = float(price_data[i].close_price or 0)

                if prev_price > 0:
                    daily_return = (curr_price - prev_price) / prev_price
                    returns.append(daily_return)

            return returns

        except Exception as e:
            logging.error(f"Error getting benchmark returns for {symbol}: {e}")
            return []

    def calculate_value_at_risk(
        self, returns: List[float], confidence_level: float = 0.05
    ) -> float:
        """Calculate Value at Risk (VaR) at given confidence level."""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        var = np.percentile(returns_array, confidence_level * 100)

        return float(-var)  # Return as positive value

    def calculate_conditional_var(
        self, returns: List[float], confidence_level: float = 0.05
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, confidence_level * 100)

        # Calculate mean of returns below VaR threshold
        tail_returns = returns_array[returns_array <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        cvar = np.mean(tail_returns)
        return float(-cvar)  # Return as positive value

    def calculate_portfolio_metrics(
        self,
        snapshots: List[PortfolioSnapshot],
        benchmark_symbol: str = "SPY",
        risk_free_rate: float = 0.02,
    ) -> Dict:
        """Calculate comprehensive portfolio metrics."""
        if len(snapshots) < 2:
            return {"error": "Insufficient data for metrics calculation"}

        # Calculate external cash flows and time-weighted portfolio returns
        try:
            from ..portfolio.manager import PortfolioManager
        except Exception:
            PortfolioManager = None

        cash_flows_by_day_float: Optional[Dict[date, float]] = None
        if PortfolioManager:
            # Best-effort: derive flows from snapshots' portfolio id via storage is not trivial here.
            # Expect caller-side computation. Fallback: treat as no flows.
            pass

        portfolio_returns = self.calculate_returns(snapshots, cash_flows_by_day_float)

        if len(portfolio_returns) == 0:
            return {"error": "Could not calculate returns"}

        # Get benchmark returns
        start_date = snapshots[0].date
        end_date = snapshots[-1].date
        benchmark_returns = self.get_benchmark_returns(
            benchmark_symbol, start_date, end_date
        )

        # Align portfolio and benchmark returns by length
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        if min_length > 0:
            portfolio_returns_aligned = portfolio_returns[-min_length:]
            benchmark_returns_aligned = benchmark_returns[-min_length:]
        else:
            portfolio_returns_aligned = portfolio_returns
            benchmark_returns_aligned = []

        metrics = {
            # Basic metrics
            "total_return": float(np.sum(portfolio_returns)),
            "annualized_return": float(np.mean(portfolio_returns) * 252),
            "volatility": self.calculate_volatility(portfolio_returns),
            "sharpe_ratio": self.calculate_sharpe_ratio(
                portfolio_returns, risk_free_rate
            ),
            "sortino_ratio": self.calculate_sortino_ratio(
                portfolio_returns, risk_free_rate=risk_free_rate
            ),
            # Risk metrics
            "max_drawdown": self.calculate_max_drawdown(snapshots)[0],
            "max_drawdown_duration": self.calculate_max_drawdown(snapshots)[1],
            "var_5pct": self.calculate_value_at_risk(portfolio_returns, 0.05),
            "cvar_5pct": self.calculate_conditional_var(portfolio_returns, 0.05),
            "calmar_ratio": self.calculate_calmar_ratio(portfolio_returns, snapshots),
            # Benchmark comparison (if available)
            "benchmark_symbol": benchmark_symbol,
            "benchmark_available": len(benchmark_returns_aligned) > 0,
        }

        # Add benchmark-relative metrics if benchmark data is available
        if len(benchmark_returns_aligned) > 0:
            metrics.update(
                {
                    "beta": self.calculate_beta(
                        portfolio_returns_aligned, benchmark_returns_aligned
                    ),
                    "alpha": self.calculate_alpha(
                        portfolio_returns_aligned,
                        benchmark_returns_aligned,
                        risk_free_rate,
                    ),
                    "information_ratio": self.calculate_information_ratio(
                        portfolio_returns_aligned, benchmark_returns_aligned
                    ),
                    "benchmark_return": float(np.mean(benchmark_returns_aligned) * 252),
                    "benchmark_volatility": self.calculate_volatility(
                        benchmark_returns_aligned
                    ),
                }
            )

        return metrics

    def calculate_sector_allocation(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate portfolio allocation by sector."""
        sector_values = {}
        total_value = 0.0

        for position in positions:
            if position.get("market_value") and position["market_value"] > 0:
                # Get instrument info to determine sector
                symbol = position["symbol"]
                instrument_info = self.data_manager.get_instrument_info(symbol)

                sector = "Unknown"
                if instrument_info and instrument_info.sector:
                    sector = instrument_info.sector

                market_value = float(position["market_value"])

                if sector in sector_values:
                    sector_values[sector] += market_value
                else:
                    sector_values[sector] = market_value

                total_value += market_value

        # Convert to percentages
        if total_value > 0:
            return {
                sector: (value / total_value) * 100
                for sector, value in sector_values.items()
            }

        return {}

    def calculate_currency_allocation(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate portfolio allocation by currency."""
        currency_values = {}
        total_value = 0.0

        for position in positions:
            if position.get("market_value") and position["market_value"] > 0:
                currency = position.get("currency", "USD")
                market_value = float(position["market_value"])

                if currency in currency_values:
                    currency_values[currency] += market_value
                else:
                    currency_values[currency] = market_value

                total_value += market_value

        # Convert to percentages
        if total_value > 0:
            return {
                currency: (value / total_value) * 100
                for currency, value in currency_values.items()
            }

        return {}
