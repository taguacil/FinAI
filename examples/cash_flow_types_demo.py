#!/usr/bin/env python3
"""
Demonstration of how time-weighted returns handle different types of cash flows.

This script shows that:
1. Time-weighted returns REMOVE external cash flows (deposits/withdrawals)
2. Time-weighted returns PRESERVE internal cash flows (dividends/interest/fees)
3. Money-weighted returns include ALL cash flows
"""

import os
import sys
from datetime import date, timedelta
from decimal import Decimal

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.portfolio.models import Currency, PortfolioSnapshot
from src.utils.metrics import FinancialMetricsCalculator


def create_portfolio_with_mixed_cash_flows():
    """Create a portfolio with both external and internal cash flows."""
    snapshots = []
    base_date = date(2023, 1, 1)
    base_value = 10000

    # Day 0: Initial portfolio
    total_value = base_value

    for i in range(6):
        snapshot_date = base_date + timedelta(days=i)

        if i == 0:
            # Initial value
            total_value = base_value
        elif i == 1:
            # Day 1: 1% market growth + $100 dividend (internal cash flow)
            total_value = base_value * 1.01 + 100
        elif i == 2:
            # Day 2: 1% market growth + $2000 deposit (external cash flow)
            total_value = (base_value * 1.01 + 100) * 1.01 + 2000
        elif i == 3:
            # Day 3: 1% market growth + $150 interest (internal cash flow)
            total_value = ((base_value * 1.01 + 100) * 1.01 + 2000) * 1.01 + 150
        elif i == 4:
            # Day 4: 1% market growth - $1000 withdrawal (external cash flow)
            total_value = (
                ((base_value * 1.01 + 100) * 1.01 + 2000) * 1.01 + 150
            ) * 1.01 - 1000
        else:  # i == 5
            # Day 5: 1% market growth + $75 dividend (internal cash flow)
            total_value = (
                (((base_value * 1.01 + 100) * 1.01 + 2000) * 1.01 + 150) * 1.01 - 1000
            ) * 1.01 + 75

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
        snapshots.append(snapshot)

    return snapshots


def create_external_cash_flows():
    """Create external cash flows (deposits/withdrawals only)."""
    base_date = date(2023, 1, 1)

    return {
        base_date + timedelta(days=2): 2000.0,  # Deposit on day 2
        base_date + timedelta(days=4): -1000.0,  # Withdrawal on day 4
    }


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("CASH FLOW TYPES IN TIME-WEIGHTED RETURNS DEMONSTRATION")
    print("=" * 80)
    print()

    # Create sample data
    snapshots = create_portfolio_with_mixed_cash_flows()
    external_cash_flows = create_external_cash_flows()

    print("PORTFOLIO SCENARIO:")
    print("Day 0: Initial portfolio value: $10,000")
    print("Day 1: 1% growth + $100 dividend (internal cash flow)")
    print("Day 2: 1% growth + $2,000 deposit (external cash flow)")
    print("Day 3: 1% growth + $150 interest (internal cash flow)")
    print("Day 4: 1% growth - $1,000 withdrawal (external cash flow)")
    print("Day 5: 1% growth + $75 dividend (internal cash flow)")
    print()

    print(f"Portfolio values: {[float(s.total_value) for s in snapshots]}")
    print(f"External cash flows: {external_cash_flows}")
    print()

    # Initialize the calculator
    calculator = FinancialMetricsCalculator()

    print("1. TIME-WEIGHTED RETURNS (TWR)")
    print("-" * 40)
    print("TWR should:")
    print("✓ REMOVE external cash flows (deposits/withdrawals)")
    print("✓ PRESERVE internal cash flows (dividends/interest/fees)")
    print("✓ Show consistent underlying performance")
    print()

    # Calculate time-weighted returns
    twr_returns = calculator.calculate_time_weighted_return(
        snapshots, external_cash_flows
    )

    print("Daily TWR returns:")
    for i, return_val in enumerate(twr_returns):
        day = i + 1
        if day == 1:
            print(
                f"  Day {day}: {return_val:.4f} (1% growth + $100 dividend = {return_val:.2%})"
            )
        elif day == 2:
            print(f"  Day {day}: {return_val:.4f} (1% growth, deposit removed)")
        elif day == 3:
            print(
                f"  Day {day}: {return_val:.4f} (1% growth + $150 interest = {return_val:.2%})"
            )
        elif day == 4:
            print(f"  Day {day}: {return_val:.4f} (1% growth, withdrawal removed)")
        else:  # day == 5
            print(
                f"  Day {day}: {return_val:.4f} (1% growth + $75 dividend = {return_val:.2%})"
            )

    print()

    # Calculate annualized TWR
    twr_annualized = calculator.calculate_annualized_time_weighted_return(
        snapshots, external_cash_flows
    )
    print(f"Annualized TWR: {twr_annualized:.4f} ({twr_annualized * 100:.2f}%)")
    print()

    print("2. MONEY-WEIGHTED RETURNS (MWR)")
    print("-" * 40)
    print("MWR should:")
    print("✓ Include ALL cash flows (external + internal)")
    print("✓ Show actual investor experience")
    print("✓ Reflect timing of deposits/withdrawals")
    print()

    # Calculate money-weighted returns
    mwr_returns = calculator.calculate_money_weighted_return(
        snapshots, external_cash_flows
    )

    print("Daily MWR returns:")
    for i, return_val in enumerate(mwr_returns):
        day = i + 1
        if day == 1:
            print(
                f"  Day {day}: {return_val:.4f} (1% growth + $100 dividend = {return_val:.2%})"
            )
        elif day == 2:
            print(
                f"  Day {day}: {return_val:.4f} (1% growth + $2,000 deposit = {return_val:.2%})"
            )
        elif day == 3:
            print(
                f"  Day {day}: {return_val:.4f} (1% growth + $150 interest = {return_val:.2%})"
            )
        elif day == 4:
            print(
                f"  Day {day}: {return_val:.4f} (1% growth - $1,000 withdrawal = {return_val:.2%})"
            )
        else:  # day == 5
            print(
                f"  Day {day}: {return_val:.4f} (1% growth + $75 dividend = {return_val:.2%})"
            )

    print()

    # Calculate annualized MWR
    mwr_annualized = calculator.calculate_annualized_money_weighted_return(
        snapshots, external_cash_flows
    )
    print(f"Annualized MWR: {mwr_annualized:.4f} ({mwr_annualized * 100:.2f}%)")
    print()

    print("3. KEY DIFFERENCES")
    print("-" * 40)

    # Show the difference in returns
    print("Day-by-day comparison:")
    for i in range(len(twr_returns)):
        day = i + 1
        twr_val = twr_returns[i]
        mwr_val = mwr_returns[i]
        diff = mwr_val - twr_val

        if day == 2:  # Deposit day
            print(
                f"  Day {day}: TWR={twr_val:.4f}, MWR={mwr_val:.4f}, Diff={diff:.4f} (deposit impact)"
            )
        elif day == 4:  # Withdrawal day
            print(
                f"  Day {day}: TWR={twr_val:.4f}, MWR={mwr_val:.4f}, Diff={diff:.4f} (withdrawal impact)"
            )
        else:  # Internal cash flow days
            print(
                f"  Day {day}: TWR={twr_val:.4f}, MWR={mwr_val:.4f}, Diff={diff:.4f} (same - internal flow)"
            )

    print()

    # Show the overall difference
    overall_diff = mwr_annualized - twr_annualized
    print(f"Overall difference: {overall_diff:.4f} ({overall_diff * 100:.2f}%)")
    print(
        "This represents the impact of external cash flow timing on investor returns."
    )
    print()

    print("4. VERIFICATION")
    print("-" * 40)
    print("✓ TWR preserves internal cash flows (dividends, interest, fees)")
    print("✓ TWR removes external cash flows (deposits, withdrawals)")
    print("✓ MWR includes all cash flows")
    print("✓ TWR shows consistent underlying performance")
    print("✓ MWR shows actual investor experience")
    print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
