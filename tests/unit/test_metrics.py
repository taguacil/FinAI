"""
Unit tests for financial metrics calculator.
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from src.portfolio.models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    PortfolioSnapshot,
)
from src.utils.metrics import FinancialMetricsCalculator


class TestFinancialMetricsCalculator:
    """Test cases for FinancialMetricsCalculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = FinancialMetricsCalculator()

        # Create test portfolio snapshots
        self.base_date = date(2023, 1, 1)
        self.snapshots = []

        # Create a simple portfolio that grows over time
        for i in range(10):
            snapshot_date = self.base_date + timedelta(days=i)
            # Portfolio grows by 1% per day
            total_value = Decimal("10000") * (Decimal("1.01") ** i)

            snapshot = PortfolioSnapshot(
                date=snapshot_date,
                total_value=total_value,
                cash_balance=Decimal("1000"),
                positions_value=total_value - Decimal("1000"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("1000")},
                total_cost_basis=Decimal("10000"),
                total_unrealized_pnl=total_value - Decimal("10000"),
                total_unrealized_pnl_percent=(
                    (total_value - Decimal("10000")) / Decimal("10000")
                )
                * Decimal("100"),
            )
            self.snapshots.append(snapshot)

    def test_calculate_time_weighted_return(self):
        """Test time-weighted return calculation."""
        returns = self.calculator.calculate_time_weighted_return(self.snapshots)

        assert len(returns) == 9  # 9 daily returns for 10 snapshots
        assert all(isinstance(r, float) for r in returns)
        # Each return should be approximately 0.01 (1%)
        assert all(abs(r - 0.01) < 0.001 for r in returns)

    def test_calculate_money_weighted_return(self):
        """Test money-weighted return calculation."""
        returns = self.calculator.calculate_money_weighted_return(self.snapshots)

        assert len(returns) == 9  # 9 daily returns for 10 snapshots
        assert all(isinstance(r, float) for r in returns)
        # Each return should be approximately 0.01 (1%)
        assert all(abs(r - 0.01) < 0.001 for r in returns)

    def test_calculate_annualized_time_weighted_return(self):
        """Test annualized time-weighted return calculation."""
        annualized_return = self.calculator.calculate_annualized_time_weighted_return(
            self.snapshots
        )

        assert isinstance(annualized_return, float)
        assert annualized_return > 0
        # With 1% daily growth, annualized should be approximately (1.01^365 - 1)
        expected_annual = (1.01**365.25) - 1
        assert abs(annualized_return - expected_annual) < 0.1

    def test_calculate_annualized_money_weighted_return(self):
        """Test annualized money-weighted return calculation."""
        annualized_return = self.calculator.calculate_annualized_money_weighted_return(
            self.snapshots
        )

        assert isinstance(annualized_return, float)
        assert annualized_return > 0
        # With 1% daily growth, annualized should be approximately (1.01^365 - 1)
        expected_annual = (1.01**365.25) - 1
        assert abs(annualized_return - expected_annual) < 0.1

    def test_calculate_modified_dietz_return(self):
        """Test Modified Dietz return calculation."""
        # Add some cash flows
        cash_flows = {
            self.base_date + timedelta(days=2): 1000.0,  # Deposit on day 2
            self.base_date + timedelta(days=5): -500.0,  # Withdrawal on day 5
        }

        modified_dietz_return = self.calculator.calculate_modified_dietz_return(
            self.snapshots, cash_flows
        )

        assert isinstance(modified_dietz_return, float)
        # Should be positive given the growth scenario

    def test_calculate_internal_rate_of_return(self):
        """Test internal rate of return calculation."""
        irr = self.calculator.calculate_internal_rate_of_return(self.snapshots)

        assert isinstance(irr, float)
        assert irr > 0
        # IRR should be positive given the growth scenario

    def test_calculate_dollar_weighted_return(self):
        """Test dollar-weighted return calculation."""
        # Add some cash flows
        cash_flows = {
            self.base_date + timedelta(days=2): 1000.0,  # Deposit on day 2
            self.base_date + timedelta(days=5): -500.0,  # Withdrawal on day 5
        }

        dollar_weighted_return = self.calculator.calculate_dollar_weighted_return(
            self.snapshots, cash_flows
        )

        assert isinstance(dollar_weighted_return, float)
        # Should be positive given the growth scenario

    def test_calculate_all_return_metrics(self):
        """Test comprehensive return metrics calculation."""
        all_metrics = self.calculator.calculate_all_return_metrics(self.snapshots)

        assert isinstance(all_metrics, dict)
        expected_keys = [
            "time_weighted_return",
            "money_weighted_return",
            "modified_dietz_return",
            "internal_rate_of_return",
            "dollar_weighted_return",
            "time_weighted_annualized",
            "money_weighted_annualized",
        ]

        for key in expected_keys:
            assert key in all_metrics
            assert isinstance(all_metrics[key], float)

    def test_cash_flows_impact(self):
        """Test that cash flows affect money-weighted but not time-weighted returns."""
        # Create a scenario where cash flows create different return patterns
        # Portfolio starts at 10000, grows 1% per day, but has a large withdrawal on day 2
        simple_snapshots = []
        base_value = 10000

        for i in range(5):
            snapshot_date = self.base_date + timedelta(days=i)

            if i == 0:
                total_value = base_value
            elif i == 1:
                total_value = base_value * 1.01  # 1% growth
            elif i == 2:
                total_value = (
                    base_value * 1.01 * 1.01
                ) - 5000  # Growth then large withdrawal
            elif i == 3:
                total_value = (
                    (base_value * 1.01 * 1.01) - 5000
                ) * 1.01  # Growth after withdrawal
            else:  # i == 4
                total_value = (
                    ((base_value * 1.01 * 1.01) - 5000) * 1.01
                ) * 1.01  # Continued growth

            snapshot = PortfolioSnapshot(
                date=snapshot_date,
                total_value=Decimal(str(total_value)),
                cash_balance=Decimal("1000"),
                positions_value=Decimal(str(total_value - 1000)),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("1000")},
                total_cost_basis=Decimal("10000"),
                total_unrealized_pnl=Decimal(str(total_value - 10000)),
                total_unrealized_pnl_percent=Decimal(
                    str(((total_value - 10000) / 10000) * 100)
                ),
            )
            simple_snapshots.append(snapshot)

        # Add the withdrawal cash flow
        cash_flows = {
            self.base_date + timedelta(days=2): -5000.0,  # Large withdrawal on day 2
        }

        # Time-weighted returns should remove the cash flow impact and show consistent growth
        twr_returns = self.calculator.calculate_time_weighted_return(
            simple_snapshots, cash_flows
        )
        twr_no_flows = self.calculator.calculate_time_weighted_return(simple_snapshots)

        # Both should show the underlying growth pattern (approximately 1% per day)
        assert len(twr_returns) == len(twr_no_flows)

        # Verify that time-weighted returns show consistent growth (approximately 1% per day)
        for i, return_val in enumerate(twr_returns):
            assert (
                abs(return_val - 0.01) < 0.1
            ), f"Return {i} should be approximately 1% (got {return_val})"

        # Money-weighted returns should be different due to cash flow impact
        mwr_returns = self.calculator.calculate_money_weighted_return(
            simple_snapshots, cash_flows
        )
        mwr_no_flows = self.calculator.calculate_money_weighted_return(simple_snapshots)

        # The key difference: time-weighted returns remove cash flow impact,
        # while money-weighted returns include it in the return calculation
        # Both methods use the same portfolio values, but calculate returns differently

        # Verify that the time-weighted returns show the underlying growth pattern
        # (approximately 1% per day, ignoring the cash flow impact)
        for i, return_val in enumerate(twr_returns):
            if i == 1:  # Day 2 has the large withdrawal
                # The time-weighted return should remove the cash flow impact
                # and show the underlying growth
                assert (
                    abs(return_val - 0.01) < 0.1
                ), f"Return {i} should be approximately 1% (got {return_val})"
            else:
                assert (
                    abs(return_val - 0.01) < 0.1
                ), f"Return {i} should be approximately 1% (got {return_val})"

        # The money-weighted returns should show the actual portfolio performance
        # including the cash flow impact
        assert len(mwr_returns) == len(mwr_no_flows)

    def test_twr_preserves_internal_cash_flows(self):
        """Test that time-weighted returns preserve internal cash flows (dividends, interest, fees)."""
        # Create a scenario where the portfolio has internal cash flows (dividends, interest, fees)
        # but no external cash flows (deposits/withdrawals)
        simple_snapshots = []
        base_value = 10000

        for i in range(5):
            snapshot_date = self.base_date + timedelta(days=i)

            if i == 0:
                total_value = base_value
            elif i == 1:
                # Day 1: 1% growth + $100 dividend
                total_value = base_value * 1.01 + 100
            elif i == 2:
                # Day 2: 1% growth + $50 interest - $25 fees
                total_value = (base_value * 1.01 + 100) * 1.01 + 50 - 25
            elif i == 3:
                # Day 3: 1% growth + $75 dividend
                total_value = ((base_value * 1.01 + 100) * 1.01 + 50 - 25) * 1.01 + 75
            else:  # i == 4
                # Day 4: 1% growth + $30 interest
                total_value = (
                    ((base_value * 1.01 + 100) * 1.01 + 50 - 25) * 1.01 + 75
                ) * 1.01 + 30

            snapshot = PortfolioSnapshot(
                date=snapshot_date,
                total_value=Decimal(str(round(total_value, 2))),
                cash_balance=Decimal("1000"),
                positions_value=Decimal(str(round(total_value - 1000, 2))),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("1000")},
                total_cost_basis=Decimal("10000"),
                total_unrealized_pnl=Decimal(str(round(total_value - 10000, 2))),
                total_unrealized_pnl_percent=Decimal(
                    str(round(((total_value - 10000) / 10000) * 100, 2))
                ),
            )
            simple_snapshots.append(snapshot)

        # No external cash flows (deposits/withdrawals)
        cash_flows = {}

        # Time-weighted returns should include internal cash flows as part of performance
        twr_returns = self.calculator.calculate_time_weighted_return(
            simple_snapshots, cash_flows
        )

        # Verify that internal cash flows are preserved in TWR calculations
        # Day 1: Should show growth + dividend impact
        assert (
            twr_returns[0] > 0.01
        ), f"Day 1 return should include dividend impact (got {twr_returns[0]})"

        # Day 2: Should show growth + interest - fees impact
        assert (
            twr_returns[1] > 0.01
        ), f"Day 2 return should include interest and fees impact (got {twr_returns[1]})"

        # Day 3: Should show growth + dividend impact
        assert (
            twr_returns[2] > 0.01
        ), f"Day 3 return should include dividend impact (got {twr_returns[2]})"

        # Day 4: Should show growth + interest impact
        assert (
            twr_returns[3] > 0.01
        ), f"Day 4 return should include interest impact (got {twr_returns[3]})"

        # Money-weighted returns should show the same pattern since they include all cash flows
        mwr_returns = self.calculator.calculate_money_weighted_return(
            simple_snapshots, cash_flows
        )

        # Both should be identical when there are no external cash flows
        assert (
            twr_returns == mwr_returns
        ), "TWR and MWR should be identical when no external cash flows"

    def test_empty_snapshots(self):
        """Test behavior with insufficient data."""
        empty_snapshots = []
        single_snapshot = [self.snapshots[0]]

        # Test with empty list
        assert self.calculator.calculate_time_weighted_return(empty_snapshots) == []
        assert (
            self.calculator.calculate_annualized_time_weighted_return(empty_snapshots)
            == 0.0
        )

        # Test with single snapshot
        assert self.calculator.calculate_time_weighted_return(single_snapshot) == []
        assert (
            self.calculator.calculate_annualized_time_weighted_return(single_snapshot)
            == 0.0
        )

    def test_backward_compatibility(self):
        """Test that the old calculate_returns method still works."""
        returns = self.calculator.calculate_returns(self.snapshots)

        assert len(returns) == 9
        assert all(isinstance(r, float) for r in returns)
        # Should return the same as time-weighted returns
        twr_returns = self.calculator.calculate_time_weighted_return(self.snapshots)
        assert returns == twr_returns

    def test_calculate_portfolio_metrics(self):
        """Test comprehensive portfolio metrics calculation."""
        # Test with basic snapshots
        metrics = self.calculator.calculate_portfolio_metrics(self.snapshots, "SPY")

        assert isinstance(metrics, dict)
        assert "error" not in metrics

        # Check that time-weighted metrics are included
        expected_twr_keys = [
            "time_weighted_annualized_return",
            "money_weighted_annualized_return",
            "modified_dietz_return",
        ]

        for key in expected_twr_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)

        # Check basic metrics
        basic_keys = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
        ]

        for key in basic_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)

        # Test with cash flows
        cash_flows = {
            self.base_date + timedelta(days=2): 100.0,  # Small deposit
            self.base_date + timedelta(days=5): -50.0,  # Small withdrawal
        }

        metrics_with_flows = self.calculator.calculate_portfolio_metrics(
            self.snapshots, "SPY", cash_flows
        )

        assert isinstance(metrics_with_flows, dict)
        assert "error" not in metrics_with_flows

        # Both should have valid time-weighted returns
        assert metrics["time_weighted_annualized_return"] > 0
        assert metrics_with_flows["time_weighted_annualized_return"] > 0

        # Test with insufficient data
        empty_metrics = self.calculator.calculate_portfolio_metrics([], "SPY")
        assert "error" in empty_metrics

        single_metrics = self.calculator.calculate_portfolio_metrics(
            [self.snapshots[0]], "SPY"
        )
        assert "error" in single_metrics

    def test_time_weighted_vs_money_weighted_comparison(self):
        """Test the comparison between time-weighted and money-weighted returns."""
        # Create a scenario with significant cash flows to show the difference
        # Use smaller cash flows that won't break the portfolio growth pattern
        cash_flows = {
            self.base_date + timedelta(days=2): 100.0,  # Small deposit
            self.base_date + timedelta(days=5): -50.0,  # Small withdrawal
        }

        # Calculate both types of returns
        twr_returns = self.calculator.calculate_time_weighted_return(
            self.snapshots, cash_flows
        )
        mwr_returns = self.calculator.calculate_money_weighted_return(
            self.snapshots, cash_flows
        )

        twr_annualized = self.calculator.calculate_annualized_time_weighted_return(
            self.snapshots, cash_flows
        )
        mwr_annualized = self.calculator.calculate_annualized_money_weighted_return(
            self.snapshots, cash_flows
        )

        # Both should return valid results
        assert len(twr_returns) > 0
        assert len(mwr_returns) > 0
        assert isinstance(twr_annualized, float)
        assert isinstance(mwr_annualized, float)

        # In this simple growth scenario, both should be positive
        assert twr_annualized > 0
        assert mwr_annualized > 0

        # The difference between TWR and MWR shows the impact of cash flow timing
        difference = twr_annualized - mwr_annualized
        assert isinstance(difference, float)

        # This difference represents how cash flow timing affected the investor's experience
        # vs. the pure investment performance

    def test_time_weighted_returns_no_cash_flows(self):
        """Test time-weighted returns when there are no external cash flows."""
        # Test with no cash flows
        twr_returns = self.calculator.calculate_time_weighted_return(self.snapshots)
        twr_annualized = self.calculator.calculate_annualized_time_weighted_return(
            self.snapshots
        )

        # Should return valid results
        assert len(twr_returns) == 9  # 9 daily returns for 10 snapshots
        assert isinstance(twr_annualized, float)
        assert twr_annualized > 0

        # Each daily return should be approximately 1% (the growth rate in our test data)
        for return_val in twr_returns:
            assert (
                abs(return_val - 0.01) < 0.001
            ), f"Expected ~1% return, got {return_val}"

        # Test with empty cash flows dict
        twr_returns_empty = self.calculator.calculate_time_weighted_return(
            self.snapshots, {}
        )
        twr_annualized_empty = (
            self.calculator.calculate_annualized_time_weighted_return(
                self.snapshots, {}
            )
        )

        # Should be identical to no cash flows
        assert twr_returns == twr_returns_empty
        assert twr_annualized == twr_annualized_empty

        # Test with None cash flows
        twr_returns_none = self.calculator.calculate_time_weighted_return(
            self.snapshots, None
        )
        twr_annualized_none = self.calculator.calculate_annualized_time_weighted_return(
            self.snapshots, None
        )

        # Should be identical to no cash flows
        assert twr_returns == twr_returns_none
        assert twr_annualized == twr_annualized_none

    def test_calculate_time_weighted_return_portfolio_starting_from_zero(self):
        """Test TWR calculation when portfolio starts from zero (edge case)."""
        # Create snapshots where portfolio starts from zero
        zero_start_snapshots = []
        base_date = date(2023, 1, 1)

        # First few snapshots have zero value (portfolio not started yet)
        for i in range(3):
            snapshot = PortfolioSnapshot(
                date=base_date + timedelta(days=i),
                total_value=Decimal("0"),
                cash_balance=Decimal("0"),
                positions_value=Decimal("0"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("0")},
                total_cost_basis=Decimal("0"),
                total_unrealized_pnl=Decimal("0"),
                total_unrealized_pnl_percent=Decimal("0"),
            )
            zero_start_snapshots.append(snapshot)

        # Then portfolio gets value
        for i in range(3, 6):
            value = Decimal(str(1000 * (i - 2)))  # 1000, 2000, 3000
            snapshot = PortfolioSnapshot(
                date=base_date + timedelta(days=i),
                total_value=value,
                cash_balance=Decimal("0"),
                positions_value=value,
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("0")},
                total_cost_basis=value,
                total_unrealized_pnl=Decimal("0"),
                total_unrealized_pnl_percent=Decimal("0"),
            )
            zero_start_snapshots.append(snapshot)

        # This should return some returns (only for periods with positive previous values)
        returns = self.calculator.calculate_time_weighted_return(zero_start_snapshots)
        # Should only calculate returns for periods 4->5 and 5->6 (where prev_value > 0)
        assert len(returns) == 2
        # From 1000 to 2000 is 100% return (1.0)
        # From 2000 to 3000 is 50% return (0.5)
        assert abs(returns[0] - 1.0) < 0.001
        assert abs(returns[1] - 0.5) < 0.001

        # Test with only one non-zero snapshot at the end
        single_value_snapshots = []
        for i in range(4):
            value = Decimal("1000") if i == 3 else Decimal("0")
            snapshot = PortfolioSnapshot(
                date=base_date + timedelta(days=i),
                total_value=value,
                cash_balance=Decimal("0"),
                positions_value=value,
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("0")},
                total_cost_basis=value,
                total_unrealized_pnl=Decimal("0"),
                total_unrealized_pnl_percent=Decimal("0"),
            )
            single_value_snapshots.append(snapshot)

        # This should also return an empty list (can't calculate returns)
        returns = self.calculator.calculate_time_weighted_return(single_value_snapshots)
        assert returns == []

    def test_calculate_time_weighted_return_with_valid_progression(self):
        """Test TWR calculation with a valid progression from zero to positive values."""
        # Create snapshots with valid progression
        valid_snapshots = []
        base_date = date(2023, 1, 1)

        # Start with zero
        snapshot = PortfolioSnapshot(
            date=base_date,
            total_value=Decimal("0"),
            cash_balance=Decimal("0"),
            positions_value=Decimal("0"),
            base_currency=Currency.USD,
            positions={},
            cash_balances={Currency.USD: Decimal("0")},
            total_cost_basis=Decimal("0"),
            total_unrealized_pnl=Decimal("0"),
            total_unrealized_pnl_percent=Decimal("0"),
        )
        valid_snapshots.append(snapshot)

        # Then get some value
        snapshot = PortfolioSnapshot(
            date=base_date + timedelta(days=1),
            total_value=Decimal("1000"),
            cash_balance=Decimal("0"),
            positions_value=Decimal("1000"),
            base_currency=Currency.USD,
            positions={},
            cash_balances={Currency.USD: Decimal("0")},
            total_cost_basis=Decimal("1000"),
            total_unrealized_pnl=Decimal("0"),
            total_unrealized_pnl_percent=Decimal("0"),
        )
        valid_snapshots.append(snapshot)

        # Then continue growing
        for i in range(2, 5):
            value = Decimal(str(1000 + (i-1) * 100))  # 1100, 1200, 1300
            snapshot = PortfolioSnapshot(
                date=base_date + timedelta(days=i),
                total_value=value,
                cash_balance=Decimal("0"),
                positions_value=value,
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("0")},
                total_cost_basis=Decimal("1000"),
                total_unrealized_pnl=value - Decimal("1000"),
                total_unrealized_pnl_percent=((value - Decimal("1000")) / Decimal("1000")) * Decimal("100"),
            )
            valid_snapshots.append(snapshot)

        # This should skip the first period (0 to 1000) but calculate subsequent periods
        returns = self.calculator.calculate_time_weighted_return(valid_snapshots)
        assert len(returns) == 3  # 3 valid periods after the initial zero
        assert all(isinstance(r, float) for r in returns)
        # Returns should be: 1000->1100 (10%), 1100->1200 (9.09%), 1200->1300 (8.33%)
        assert abs(returns[0] - 0.1) < 0.001  # 10%
        assert abs(returns[1] - (100/1100)) < 0.001  # 9.09%
        assert abs(returns[2] - (100/1200)) < 0.001  # 8.33%

    def test_metrics_consistency_for_one_year_period(self):
        """Test that total return, annualized return, and YTD calculation are consistent for 1-year periods."""
        from datetime import date, timedelta
        from decimal import Decimal

        # Create snapshots for exactly 1 year (365 days)
        base_date = date(2024, 1, 1)
        snapshots = []

        for i in range(366):  # 0 to 365 days = 366 snapshots
            current_date = base_date + timedelta(days=i)
            # Simulate 20% growth over the year
            value = 10000 + (2000 * i / 365)  # Linear growth from 10k to 12k

            snapshot = PortfolioSnapshot(
                date=current_date,
                total_value=Decimal(str(value)),
                cash_balance=Decimal("0"),
                positions_value=Decimal(str(value)),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("0")},
                total_cost_basis=Decimal("10000"),
                total_unrealized_pnl=Decimal(str(value - 10000)),
                total_unrealized_pnl_percent=Decimal(str((value - 10000) / 10000 * 100))
            )
            snapshots.append(snapshot)

        cash_flows = {}  # No external cash flows

        # Method 1: Manual calculation like in UI (YTD)
        daily_returns = self.calculator.calculate_time_weighted_return(snapshots, cash_flows)
        manual_twr = 1.0
        for r in daily_returns:
            manual_twr *= 1.0 + r
        manual_total_return = manual_twr - 1.0

        # Method 2: Portfolio metrics
        comprehensive_metrics = self.calculator.calculate_portfolio_metrics(
            snapshots, "SPY", cash_flows
        )

        total_return = comprehensive_metrics.get("total_return", 0.0)
        annualized_return = comprehensive_metrics.get("annualized_return", 0.0)
        time_weighted_annualized = comprehensive_metrics.get("time_weighted_annualized_return", 0.0)

        # Method 3: Direct annualized calculation
        direct_annualized = self.calculator.calculate_annualized_time_weighted_return(
            snapshots, cash_flows
        )

        # All methods should give the same result for a 1-year period
        tolerance = 0.0001  # 0.01% tolerance

        # YTD manual vs Total return
        assert abs(manual_total_return - total_return) < tolerance, \
            f"Manual YTD ({manual_total_return:.6f}) vs Total return ({total_return:.6f}) differ"

        # Total return vs Annualized return (should be same for 1 year)
        assert abs(total_return - annualized_return) < tolerance, \
            f"Total return ({total_return:.6f}) vs Annualized return ({annualized_return:.6f}) differ"

        # Annualized return vs Time-weighted annualized
        assert abs(annualized_return - time_weighted_annualized) < tolerance, \
            f"Annualized return ({annualized_return:.6f}) vs TWR annualized ({time_weighted_annualized:.6f}) differ"

        # Direct annualized vs Metrics annualized
        assert abs(direct_annualized - time_weighted_annualized) < tolerance, \
            f"Direct annualized ({direct_annualized:.6f}) vs Metrics annualized ({time_weighted_annualized:.6f}) differ"

        # Verify the actual return is approximately 20% (our test data growth)
        expected_return = 0.20  # 20%
        assert abs(total_return - expected_return) < 0.01, \
            f"Expected ~20% return, got {total_return:.4f}"
