"""
Financial metrics calculator for portfolio analysis.

This module provides comprehensive portfolio performance analysis with multiple return calculation
methodologies:

1. Time-Weighted Returns (TWR):
   - calculate_time_weighted_return(): Daily returns ignoring external cash flows
   - calculate_annualized_time_weighted_return(): Annualized TWR using geometric linking
   - Best for measuring investment manager performance (eliminates cash flow impact)

2. Money-Weighted Returns (MWR):
   - calculate_money_weighted_return(): Daily returns including cash flows
   - calculate_annualized_money_weighted_return(): Annualized MWR
   - calculate_internal_rate_of_return(): IRR using Newton-Raphson method
   - calculate_dollar_weighted_return(): Dollar-weighted return considering flow timing
   - Best for measuring investor's actual experience

3. Hybrid Approaches:
   - calculate_modified_dietz_return(): Time-weighted with cash flow timing consideration
   - Provides balance between TWR and MWR methodologies

4. Comprehensive Analysis:
   - calculate_all_return_metrics(): All return metrics in one call
   - calculate_portfolio_metrics(): Complete portfolio analysis including returns

Note: Time-weighted returns are the industry standard for performance measurement as they
eliminate the distorting effects of EXTERNAL cash flows (deposits/withdrawals), while
preserving INTERNAL cash flows (dividends, interest, fees) as part of investment performance.
Money-weighted returns show the actual return experienced by the investor including all cash flows.
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

    def calculate_time_weighted_return(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
    ) -> List[float]:
        """Calculate time-weighted returns (TWR) ignoring external cash flows.

        Time-weighted returns measure the compound rate of growth of one unit of currency
        invested in the portfolio. This method eliminates the impact of external cash flows
        (deposits/withdrawals) on performance measurement, but preserves internal cash flows
        (dividends, interest, fees) as part of the investment performance.

        Args:
            snapshots: List of portfolio snapshots
            cash_flows_by_day: Optional dict mapping dates to cash flows (positive for deposits, negative for withdrawals)
                             Note: This should only contain DEPOSIT and WITHDRAWAL transactions, not DIVIDEND, INTEREST, or FEES

        Returns:
            List of daily time-weighted returns
        """
        if len(snapshots) < 2:
            return []

        returns = []
        for i in range(1, len(snapshots)):
            prev_value = float(snapshots[i - 1].total_value)
            curr_value = float(snapshots[i].total_value)

            # Extract external cash flow for this day if provided
            # Note: cash_flows_by_day should only contain deposits/withdrawals, not dividends/interest/fees
            external_cf = 0.0
            if cash_flows_by_day:
                external_cf = float(cash_flows_by_day.get(snapshots[i].date, 0.0))

            # TWR formula: (V_t - V_{t-1} - External_CF_t) / V_{t-1}
            # This removes ONLY external cash flows (deposits/withdrawals) from return calculation
            # Internal cash flows (dividends, interest, fees) remain as part of the portfolio performance
            if prev_value > 0:
                daily_return = (curr_value - prev_value - external_cf) / prev_value
                returns.append(daily_return)

        return returns

    def calculate_money_weighted_return(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
    ) -> List[float]:
        """Calculate money-weighted returns (MWR) including external cash flows.

        Money-weighted returns measure the internal rate of return considering the timing
        and magnitude of cash flows. This method includes the impact of deposits and
        withdrawals on performance measurement, while naturally including internal cash flows
        (dividends, interest, fees) as part of the portfolio value changes.

        Args:
            snapshots: List of portfolio snapshots
            cash_flows_by_day: Optional dict mapping dates to cash flows (positive for deposits, negative for withdrawals)
                             Note: This should only contain DEPOSIT and WITHDRAWAL transactions, not DIVIDEND, INTEREST, or FEES

        Returns:
            List of daily money-weighted returns
        """
        if len(snapshots) < 2:
            return []

        returns = []
        for i in range(1, len(snapshots)):
            prev_value = float(snapshots[i - 1].total_value)
            curr_value = float(snapshots[i].total_value)

            # Note: For MWR, we don't need to extract cash flows from the parameter
            # because we want to include ALL cash flow impacts in the return calculation
            # The portfolio snapshots' total_value already includes the impact of:
            # - External flows (deposits/withdrawals)
            # - Internal flows (dividends, interest, fees)
            # - Market price changes

            # MWR formula: (V_t - V_{t-1}) / V_{t-1}
            # This includes the impact of ALL cash flows in return calculation
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)

        return returns

    def calculate_annualized_time_weighted_return(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
    ) -> float:
        """Calculate annualized time-weighted return.

        This method computes the compound annual growth rate (CAGR) using time-weighted
        returns, which is the standard for measuring portfolio performance as it eliminates
        the distorting effects of cash flows.

        Args:
            snapshots: List of portfolio snapshots
            cash_flows_by_day: Optional dict mapping dates to cash flows

        Returns:
            Annualized time-weighted return as a decimal
        """
        if len(snapshots) < 2:
            return 0.0

        # Calculate time-weighted returns (ignoring cash flows)
        twr_returns = self.calculate_time_weighted_return(snapshots, cash_flows_by_day)

        if not twr_returns:
            return 0.0

        # Calculate the total period return using geometric linking
        # (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        total_return = 1.0
        for daily_return in twr_returns:
            total_return *= 1 + daily_return
        total_return -= 1.0

        # Calculate the number of years between first and last snapshot
        start_date = snapshots[0].date
        end_date = snapshots[-1].date
        days = (end_date - start_date).days
        years = days / 365.25

        if years <= 0:
            return 0.0

        # Annualize using the formula: (1 + total_return)^(1/years) - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1

        return float(annualized_return)

    def calculate_annualized_money_weighted_return(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
    ) -> float:
        """Calculate annualized money-weighted return.

        This method computes the internal rate of return (IRR) considering cash flows,
        which shows the actual return experienced by the investor including the timing
        of deposits and withdrawals.

        Args:
            snapshots: List of portfolio snapshots
            cash_flows_by_day: Optional dict mapping dates to cash flows

        Returns:
            Annualized money-weighted return as a decimal
        """
        if len(snapshots) < 2:
            return 0.0

        # Calculate money-weighted returns (including cash flows)
        mwr_returns = self.calculate_money_weighted_return(snapshots, cash_flows_by_day)

        if not mwr_returns:
            return 0.0

        # Calculate the total period return using geometric linking
        total_return = 1.0
        for daily_return in mwr_returns:
            total_return *= 1 + daily_return
        total_return -= 1.0

        # Calculate the number of years between first and last snapshot
        start_date = snapshots[0].date
        end_date = snapshots[-1].date
        days = (end_date - start_date).days
        years = days / 365.25

        if years <= 0:
            return 0.0

        # Annualize using the formula: (1 + total_return)^(1/years) - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1

        return float(annualized_return)

    def calculate_modified_dietz_return(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
    ) -> float:
        """Calculate Modified Dietz return, a time-weighted approach that considers cash flow timing.

        The Modified Dietz method weights cash flows by the amount of time they have been
        invested in the portfolio, providing a more accurate measure than simple returns
        when there are significant cash flows.

        Args:
            snapshots: List of portfolio snapshots
            cash_flows_by_day: Optional dict mapping dates to cash flows (positive for deposits, negative for withdrawals)

        Returns:
            Modified Dietz return as a decimal
        """
        if len(snapshots) < 2:
            return 0.0

        start_date = snapshots[0].date
        end_date = snapshots[-1].date
        total_days = (end_date - start_date).days

        if total_days <= 0:
            return 0.0

        # Initial portfolio value
        initial_value = float(snapshots[0].total_value)

        # Final portfolio value
        final_value = float(snapshots[-1].total_value)

        # Calculate weighted cash flows
        weighted_cash_flows = 0.0
        if cash_flows_by_day:
            for flow_date, flow_amount in cash_flows_by_day.items():
                # Calculate days from start to cash flow
                days_from_start = (flow_date - start_date).days
                if 0 <= days_from_start <= total_days:
                    # Weight cash flow by time invested
                    weight = (total_days - days_from_start) / total_days
                    weighted_cash_flows += float(flow_amount) * weight

        # Modified Dietz formula: (End - Begin - Weighted Cash Flows) / (Begin + Weighted Cash Flows)
        if initial_value + weighted_cash_flows == 0:
            return 0.0

        modified_dietz_return = (final_value - initial_value - weighted_cash_flows) / (
            initial_value + weighted_cash_flows
        )

        return float(modified_dietz_return)

    def calculate_internal_rate_of_return(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
    ) -> float:
        """Calculate Internal Rate of Return (IRR) using Newton-Raphson method.

        IRR is the discount rate that makes the net present value of all cash flows
        (including initial investment and final value) equal to zero. This is the
        most accurate money-weighted return measure.

        Args:
            snapshots: List of portfolio snapshots
            cash_flows_by_day: Optional dict mapping dates to cash flows
            tolerance: Convergence tolerance for IRR calculation
            max_iterations: Maximum number of iterations for convergence

        Returns:
            Internal rate of return as a decimal, or 0.0 if calculation fails
        """
        if len(snapshots) < 2:
            return 0.0

        start_date = snapshots[0].date
        end_date = snapshots[-1].date
        total_days = (end_date - start_date).days

        if total_days <= 0:
            return 0.0

        # Initial portfolio value (negative as it's an outflow)
        initial_value = -float(snapshots[0].total_value)

        # Final portfolio value (positive as it's an inflow)
        final_value = float(snapshots[-1].total_value)

        # Prepare cash flows with dates
        cash_flows = [(0, initial_value)]  # Day 0: initial investment

        if cash_flows_by_day:
            for flow_date, flow_amount in cash_flows_by_day.items():
                days_from_start = (flow_date - start_date).days
                if 0 <= days_from_start <= total_days:
                    cash_flows.append((days_from_start, float(flow_amount)))

        # Add final value
        cash_flows.append((total_days, final_value))

        # Sort by day
        cash_flows.sort(key=lambda x: x[0])

        # Newton-Raphson method to find IRR
        # Start with a reasonable guess
        rate = 0.1  # 10% initial guess

        for iteration in range(max_iterations):
            npv = 0.0
            npv_derivative = 0.0

            for day, amount in cash_flows:
                # Convert days to years
                years = day / 365.25

                # Calculate present value
                if years == 0:
                    npv += amount
                else:
                    present_value = amount / ((1 + rate) ** years)
                    npv += present_value

                    # Calculate derivative for Newton-Raphson
                    if rate != -1:  # Avoid division by zero
                        npv_derivative -= (years * amount) / ((1 + rate) ** (years + 1))

            # Check convergence
            if abs(npv) < tolerance:
                break

            # Update rate using Newton-Raphson formula
            if abs(npv_derivative) > tolerance:
                rate = rate - npv / npv_derivative
            else:
                # If derivative is too small, try a different approach
                rate = rate * 1.1

            # Ensure rate stays reasonable
            if rate < -0.99 or rate > 10:
                rate = 0.1  # Reset to reasonable value

        # Convert to annual rate
        annual_irr = (1 + rate) ** (365.25 / total_days) - 1 if total_days > 0 else 0.0

        return float(annual_irr)

    def calculate_dollar_weighted_return(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
    ) -> float:
        """Calculate dollar-weighted return, considering the size and timing of cash flows.

        Dollar-weighted return measures the actual return experienced by the investor,
        taking into account when money was invested or withdrawn and in what amounts.
        This method is particularly useful for portfolios with significant cash flows.

        Args:
            snapshots: List of portfolio snapshots
            cash_flows_by_day: Optional dict mapping dates to cash flows

        Returns:
            Dollar-weighted return as a decimal
        """
        if len(snapshots) < 2:
            return 0.0

        start_date = snapshots[0].date
        end_date = snapshots[-1].date
        total_days = (end_date - start_date).days

        if total_days <= 0:
            return 0.0

        # Initial portfolio value
        initial_value = float(snapshots[0].total_value)

        # Final portfolio value
        final_value = float(snapshots[-1].total_value)

        # Calculate total cash flows and their timing
        total_cash_flows = 0.0
        weighted_cash_flows = 0.0

        if cash_flows_by_day:
            for flow_date, flow_amount in cash_flows_by_day.items():
                days_from_start = (flow_date - start_date).days
                if 0 <= days_from_start <= total_days:
                    total_cash_flows += float(flow_amount)
                    # Weight by time invested (more weight for earlier flows)
                    weight = (total_days - days_from_start) / total_days
                    weighted_cash_flows += float(flow_amount) * weight

        # Calculate average capital employed
        # This considers both initial investment and timing of cash flows
        average_capital = initial_value + weighted_cash_flows

        if average_capital == 0:
            return 0.0

        # Dollar-weighted return formula
        # (Final Value - Initial Value - Total Cash Flows) / Average Capital
        dollar_weighted_return = (
            final_value - initial_value - total_cash_flows
        ) / average_capital

        return float(dollar_weighted_return)

    def calculate_all_return_metrics(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
    ) -> Dict[str, float]:
        """Calculate all return metrics for comprehensive portfolio analysis.

        This method provides a complete view of portfolio performance using different
        methodologies, allowing users to understand both the investment manager's
        performance (time-weighted) and their personal experience (money-weighted).

        Args:
            snapshots: List of portfolio snapshots
            cash_flows_by_day: Optional dict mapping dates to cash flows

        Returns:
            Dictionary containing all return metrics
        """
        if len(snapshots) < 2:
            return {
                "error": "Insufficient data for return calculation",
                "time_weighted_return": 0.0,
                "money_weighted_return": 0.0,
                "modified_dietz_return": 0.0,
                "internal_rate_of_return": 0.0,
                "dollar_weighted_return": 0.0,
                "time_weighted_annualized": 0.0,
                "money_weighted_annualized": 0.0,
            }

        # Calculate all return metrics
        metrics = {
            "time_weighted_return": self.calculate_annualized_time_weighted_return(
                snapshots, cash_flows_by_day
            ),
            "money_weighted_return": self.calculate_annualized_money_weighted_return(
                snapshots, cash_flows_by_day
            ),
            "modified_dietz_return": self.calculate_modified_dietz_return(
                snapshots, cash_flows_by_day
            ),
            "internal_rate_of_return": self.calculate_internal_rate_of_return(
                snapshots, cash_flows_by_day
            ),
            "dollar_weighted_return": self.calculate_dollar_weighted_return(
                snapshots, cash_flows_by_day
            ),
        }

        # Add annualized versions
        metrics["time_weighted_annualized"] = metrics["time_weighted_return"]
        metrics["money_weighted_annualized"] = metrics["money_weighted_return"]

        return metrics

    def _calculate_total_return_from_daily_returns(self, daily_returns: List[float]) -> float:
        """Calculate total return from daily returns using geometric linking."""
        if not daily_returns:
            return 0.0

        # Use geometric linking: (1+r1)*(1+r2)*...*(1+rn) - 1
        total_return = 1.0
        for daily_return in daily_returns:
            total_return *= (1 + daily_return)
        total_return -= 1.0

        return float(total_return)

    def _calculate_annualized_return_from_daily_returns(self, daily_returns: List[float], snapshots: List[PortfolioSnapshot]) -> float:
        """Calculate annualized return from daily returns using proper geometric annualization."""
        if not daily_returns or len(snapshots) < 2:
            return 0.0

        # Calculate total return using geometric linking
        total_return = self._calculate_total_return_from_daily_returns(daily_returns)

        # Calculate the number of years between first and last snapshot
        start_date = snapshots[0].date
        end_date = snapshots[-1].date
        days = (end_date - start_date).days
        years = days / 365.25

        if years <= 0:
            return 0.0

        # Annualize using the formula: (1 + total_return)^(1/years) - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1

        return float(annualized_return)

    def calculate_returns(
        self,
        snapshots: List[PortfolioSnapshot],
        cash_flows_by_day: Optional[Dict[date, float]] = None,
    ) -> List[float]:
        """Calculate daily time-weighted returns (ignoring external cash injections/withdrawals).

        If cash_flows_by_day is provided (base currency amounts, positive for deposits,
        negative for withdrawals), the return for day t is computed as:
        r_t = (V_t - V_{t-1} - CF_t) / V_{t-1}, where CF_t is external cash flow on day t.

        Note: This method is now an alias for calculate_time_weighted_return for backward compatibility.
        """
        return self.calculate_time_weighted_return(snapshots, cash_flows_by_day)

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
        cash_flows_by_day: Optional[Dict[date, float]] = None,
        risk_free_rate: float = 0.02,
    ) -> Dict:
        """Calculate comprehensive portfolio metrics."""
        if len(snapshots) < 2:
            return {"error": "Insufficient data for metrics calculation"}

        # Use provided cash flows or try to derive them
        cash_flows_by_day_float: Optional[Dict[date, float]] = cash_flows_by_day
        if cash_flows_by_day_float is None:
            try:
                from ..portfolio.manager import PortfolioManager
            except Exception:
                PortfolioManager = None

            if PortfolioManager:
                # Best-effort: derive flows from snapshots' portfolio id via storage is not trivial here.
                # Expect caller-side computation. Fallback: treat as no flows.
                pass

        # Calculate both time-weighted and money-weighted returns
        portfolio_returns_twr = self.calculate_time_weighted_return(
            snapshots, cash_flows_by_day_float
        )
        portfolio_returns_mwr = self.calculate_money_weighted_return(
            snapshots, cash_flows_by_day_float
        )

        # Use time-weighted returns for backward compatibility and standard performance metrics
        portfolio_returns = portfolio_returns_twr

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
            # Basic metrics - Fixed to use proper geometric linking
            "total_return": self._calculate_total_return_from_daily_returns(portfolio_returns),
            "annualized_return": self._calculate_annualized_return_from_daily_returns(portfolio_returns, snapshots),
            "volatility": self.calculate_volatility(portfolio_returns),
            "sharpe_ratio": self.calculate_sharpe_ratio(
                portfolio_returns, risk_free_rate
            ),
            "sortino_ratio": self.calculate_sortino_ratio(
                portfolio_returns, risk_free_rate=risk_free_rate
            ),
            # Return calculation methods
            "time_weighted_annualized_return": self.calculate_annualized_time_weighted_return(
                snapshots, cash_flows_by_day_float
            ),
            "money_weighted_annualized_return": self.calculate_annualized_money_weighted_return(
                snapshots, cash_flows_by_day_float
            ),
            "modified_dietz_return": self.calculate_modified_dietz_return(
                snapshots, cash_flows_by_day_float
            ),
            "internal_rate_of_return": self.calculate_internal_rate_of_return(
                snapshots, cash_flows_by_day_float
            ),
            "dollar_weighted_return": self.calculate_dollar_weighted_return(
                snapshots, cash_flows_by_day_float
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
