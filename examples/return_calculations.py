#!/usr/bin/env python3
"""
Example script demonstrating time-weighted vs money-weighted return calculations.

This script shows how to use the new return calculation methods in the FinancialMetricsCalculator
to analyze portfolio performance with and without considering cash flows.
"""

import sys
import os
from datetime import date, timedelta
from decimal import Decimal

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.metrics import FinancialMetricsCalculator
from src.portfolio.models import PortfolioSnapshot, Currency


def create_sample_portfolio_snapshots():
    """Create sample portfolio snapshots for demonstration."""
    snapshots = []
    base_date = date(2023, 1, 1)
    base_value = 10000

    # Create a portfolio that grows over time with some volatility
    for i in range(30):  # 30 days of data
        snapshot_date = base_date + timedelta(days=i)

        # Simulate more realistic market performance with moderate growth
        if i == 0:
            total_value = base_value
        elif i == 5:
            # Large deposit on day 5
            total_value = base_value * (1.001 ** i) + 2000
        elif i == 15:
            # Large withdrawal on day 15
            total_value = base_value * (1.001 ** i) - 1500
        elif i == 20:
            # Another deposit on day 20
            total_value = base_value * (1.001 ** i) + 1000
        else:
            # Normal growth with some volatility (0.1% daily growth on average)
            growth_rate = 1.001 + (0.0005 * (i % 3 - 1))  # Varies between 0.9995 and 1.0015
            total_value = base_value * (growth_rate ** i)

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
            total_unrealized_pnl_percent=Decimal(str(round(((total_value - 10000) / 10000) * 100, 2)))
        )
        snapshots.append(snapshot)

    return snapshots


def create_cash_flows():
    """Create sample cash flows for demonstration."""
    base_date = date(2023, 1, 1)

    return {
        base_date + timedelta(days=5): 2000.0,   # Deposit on day 5
        base_date + timedelta(days=15): -1500.0,  # Withdrawal on day 15
        base_date + timedelta(days=20): 1000.0,   # Deposit on day 20
    }


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("PORTFOLIO RETURN CALCULATION DEMONSTRATION")
    print("=" * 80)
    print()

    # Create sample data
    snapshots = create_sample_portfolio_snapshots()
    cash_flows = create_cash_flows()

    print(f"Created {len(snapshots)} portfolio snapshots from {snapshots[0].date} to {snapshots[-1].date}")
    print(f"Cash flows: {len(cash_flows)} transactions")
    print()

    # Initialize the calculator
    calculator = FinancialMetricsCalculator()

    print("1. TIME-WEIGHTED RETURNS (TWR)")
    print("-" * 40)
    print("Time-weighted returns measure investment manager performance by eliminating")
    print("the impact of external cash flows (deposits/withdrawals).")
    print()

    # Calculate time-weighted returns
    twr_daily = calculator.calculate_time_weighted_return(snapshots, cash_flows)
    twr_annualized = calculator.calculate_annualized_time_weighted_return(snapshots, cash_flows)

    print(f"Daily TWR returns: {len(twr_daily)} periods")
    print(f"Average daily TWR: {sum(twr_daily) / len(twr_daily):.4f}")
    print(f"Annualized TWR: {twr_annualized:.4f} ({twr_annualized * 100:.2f}%)")
    print()

    print("2. MONEY-WEIGHTED RETURNS (MWR)")
    print("-" * 40)
    print("Money-weighted returns measure the actual return experienced by the investor,")
    print("including the timing and magnitude of cash flows.")
    print()

    # Calculate money-weighted returns
    mwr_daily = calculator.calculate_money_weighted_return(snapshots, cash_flows)
    mwr_annualized = calculator.calculate_annualized_money_weighted_return(snapshots, cash_flows)

    print(f"Daily MWR returns: {len(mwr_daily)} periods")
    print(f"Average daily MWR: {sum(mwr_daily) / len(mwr_daily):.4f}")
    print(f"Annualized MWR: {mwr_annualized:.4f} ({mwr_annualized * 100:.2f}%)")
    print()

    print("3. ALTERNATIVE RETURN MEASURES")
    print("-" * 40)

    # Calculate other return measures
    modified_dietz = calculator.calculate_modified_dietz_return(snapshots, cash_flows)
    irr = calculator.calculate_internal_rate_of_return(snapshots, cash_flows)
    dollar_weighted = calculator.calculate_dollar_weighted_return(snapshots, cash_flows)

    print(f"Modified Dietz Return: {modified_dietz:.4f} ({modified_dietz * 100:.2f}%)")
    print(f"Internal Rate of Return (IRR): {irr:.4f} ({irr * 100:.2f}%)")
    print(f"Dollar-Weighted Return: {dollar_weighted:.4f} ({dollar_weighted * 100:.2f}%)")
    print()

    print("4. COMPREHENSIVE ANALYSIS")
    print("-" * 40)

    # Get all return metrics at once
    all_metrics = calculator.calculate_all_return_metrics(snapshots, cash_flows)

    print("All Return Metrics:")
    for key, value in all_metrics.items():
        if key != "error":
            print(f"  {key}: {value:.4f} ({value * 100:.2f}%)")
    print()

    print("5. KEY DIFFERENCES")
    print("-" * 40)
    print("• Time-Weighted Returns (TWR):")
    print("  - Eliminate cash flow impact")
    print("  - Measure investment manager performance")
    print("  - Industry standard for performance measurement")
    print("  - Annualized: {:.2f}%".format(twr_annualized * 100))
    print()

    print("• Money-Weighted Returns (MWR):")
    print("  - Include cash flow impact")
    print("  - Measure investor's actual experience")
    print("  - Reflect timing of deposits/withdrawals")
    print("  - Annualized: {:.2f}%".format(mwr_annualized * 100))
    print()

    print("• Difference: {:.2f}%".format((twr_annualized - mwr_annualized) * 100))
    print("  (This shows the impact of cash flow timing on investor returns)")
    print()

    print("6. WHEN TO USE EACH METHOD")
    print("-" * 40)
    print("• Use TWR when:")
    print("  - Evaluating investment manager performance")
    print("  - Comparing different investment strategies")
    print("  - Reporting to clients or regulators")
    print()

    print("• Use MWR when:")
    print("  - Understanding personal investment experience")
    print("  - Planning future cash flows")
    print("  - Evaluating investment decisions including timing")
    print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
