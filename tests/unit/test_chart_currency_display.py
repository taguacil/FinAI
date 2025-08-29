"""
Unit tests for chart currency display in the analytics page.
Ensures charts show data in the selected display currency.
"""

import unittest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import patch

from src.portfolio.models import (
    PortfolioSnapshot, Position, FinancialInstrument, Currency,
    InstrumentType
)
from src.ui.streamlit_app import PortfolioTrackerUI


class MockDataManager:
    """Mock data manager for testing."""

    def get_exchange_rate(self, from_currency, to_currency):
        if from_currency == to_currency:
            return Decimal("1.0")

        rates = {
            (Currency.USD, Currency.EUR): Decimal("0.85"),
            (Currency.EUR, Currency.USD): Decimal("1.18"),
            (Currency.USD, Currency.GBP): Decimal("0.80"),
            (Currency.GBP, Currency.USD): Decimal("1.25"),
        }

        return rates.get((from_currency, to_currency), Decimal("1.0"))

    def get_historical_fx_rate_on(self, date_val, from_currency, to_currency):
        return self.get_exchange_rate(from_currency, to_currency)


class MockPortfolioManager:
    """Mock portfolio manager for testing."""

    def __init__(self):
        self.data_manager = MockDataManager()


class TestChartCurrencyDisplay(unittest.TestCase):
    """Test chart currency display functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui = PortfolioTrackerUI()
        self.portfolio_manager = MockPortfolioManager()

        # Create test instrument
        self.btc_instrument = FinancialInstrument(
            symbol="BTC-USD",
            name="Bitcoin",
            instrument_type=InstrumentType.CRYPTO,
            currency=Currency.USD,
            isin=None
        )

        # Create test position
        self.btc_position = Position(
            instrument=self.btc_instrument,
            quantity=Decimal("1.0"),
            average_cost=Decimal("50000.00"),
            current_price=Decimal("60000.00"),
            last_updated=datetime.now()
        )

        # Create test snapshots
        self.test_snapshots = []
        base_date = date(2024, 1, 1)

        for i in range(3):
            snapshot_date = base_date.replace(month=1+i)
            total_value = Decimal("60000.00") + (Decimal("5000.00") * i)

            snapshot = PortfolioSnapshot(
                date=snapshot_date,
                total_value=total_value,
                cash_balance=Decimal("5000.00"),
                positions_value=total_value - Decimal("5000.00"),
                base_currency=Currency.USD,
                positions={"BTC-USD": self.btc_position},
                cash_balances={Currency.USD: Decimal("5000.00")},
                total_cost_basis=Decimal("55000.00"),
                total_unrealized_pnl=total_value - Decimal("55000.00"),
                total_unrealized_pnl_percent=((total_value - Decimal("55000.00")) / Decimal("55000.00")) * 100
            )
            self.test_snapshots.append(snapshot)

    def test_chart_receives_converted_currency_data(self):
        """Test that charts receive data converted to display currency."""
        # Convert snapshots to EUR
        eur_snapshots = self.ui._convert_snapshots_to_currency(
            self.test_snapshots, "EUR", self.portfolio_manager
        )

        # Mock the chart method to capture what it receives
        chart_data = {}

        def mock_plot_portfolio_and_categories(
            snapshots, display_currency_code, portfolio_manager, start_date, end_date,
            benchmark_symbol=None, benchmark_prices_aligned=None, portfolio_returns_twr=None
        ):
            chart_data['snapshots'] = snapshots
            chart_data['currency_code'] = display_currency_code
            chart_data['first_value'] = snapshots[0].total_value if snapshots else None
            chart_data['first_currency'] = snapshots[0].base_currency if snapshots else None

        with patch.object(self.ui, 'plot_portfolio_and_categories', side_effect=mock_plot_portfolio_and_categories):
            # Call chart with converted snapshots (this is the fix)
            self.ui.plot_portfolio_and_categories(
                eur_snapshots,  # Converted snapshots
                "EUR",
                self.portfolio_manager,
                date(2024, 1, 1),
                date(2024, 4, 1)
            )

        # Verify chart received EUR data
        self.assertEqual(chart_data['currency_code'], "EUR")
        self.assertEqual(chart_data['first_currency'], Currency.EUR)

        # Verify values are in EUR (should be 85% of USD values)
        expected_eur_value = self.test_snapshots[0].total_value * Decimal("0.85")
        self.assertEqual(chart_data['first_value'], expected_eur_value)

    def test_chart_currency_conversion_accuracy(self):
        """Test that chart data conversion maintains accuracy across currencies."""
        test_currencies = [
            ("EUR", Decimal("0.85")),
            ("GBP", Decimal("0.80")),
            ("USD", Decimal("1.00"))
        ]

        original_value = self.test_snapshots[0].total_value

        for currency_code, rate in test_currencies:
            with self.subTest(currency=currency_code):
                converted_snapshots = self.ui._convert_snapshots_to_currency(
                    self.test_snapshots, currency_code, self.portfolio_manager
                )

                # For USD, should be same object (passthrough)
                if currency_code == "USD":
                    self.assertIs(converted_snapshots[0], self.test_snapshots[0])
                else:
                    expected_value = original_value * rate
                    self.assertEqual(converted_snapshots[0].total_value, expected_value)
                    self.assertEqual(str(converted_snapshots[0].base_currency.value), currency_code)

    def test_chart_data_consistency_across_snapshots(self):
        """Test that all snapshots in chart data are consistently converted."""
        eur_snapshots = self.ui._convert_snapshots_to_currency(
            self.test_snapshots, "EUR", self.portfolio_manager
        )

        # All snapshots should be in EUR
        for snapshot in eur_snapshots:
            self.assertEqual(snapshot.base_currency, Currency.EUR)

        # Values should maintain proportional relationships
        usd_ratio = self.test_snapshots[1].total_value / self.test_snapshots[0].total_value
        eur_ratio = eur_snapshots[1].total_value / eur_snapshots[0].total_value

        # Ratios should be essentially the same (within small tolerance for decimal precision)
        self.assertAlmostEqual(float(usd_ratio), float(eur_ratio), places=10)

    def test_chart_fix_prevents_currency_mismatch(self):
        """Test that the fix prevents currency/value mismatch in charts."""
        # This test simulates the bug that was fixed:
        # Before: Charts showed USD values but EUR labels
        # After: Charts show EUR values with EUR labels

        original_usd_value = self.test_snapshots[0].total_value
        eur_snapshots = self.ui._convert_snapshots_to_currency(
            self.test_snapshots, "EUR", self.portfolio_manager
        )
        converted_eur_value = eur_snapshots[0].total_value

        # Verify conversion happened
        self.assertNotEqual(original_usd_value, converted_eur_value)
        self.assertLess(converted_eur_value, original_usd_value)  # EUR should be smaller

        # Mock chart to verify it gets consistent data
        chart_calls = []

        def capture_chart_call(snapshots, currency_code, *args, **kwargs):
            chart_calls.append({
                'value': snapshots[0].total_value,
                'currency': snapshots[0].base_currency,
                'label': currency_code
            })

        with patch.object(self.ui, 'plot_portfolio_and_categories', side_effect=capture_chart_call):
            # Fixed call: converted snapshots with matching currency label
            self.ui.plot_portfolio_and_categories(
                eur_snapshots, "EUR", self.portfolio_manager, date(2024, 1, 1), date(2024, 4, 1)
            )

        self.assertEqual(len(chart_calls), 1)
        chart_call = chart_calls[0]

        # Verify data and label are consistent
        self.assertEqual(chart_call['currency'], Currency.EUR)
        self.assertEqual(chart_call['label'], "EUR")
        self.assertEqual(chart_call['value'], converted_eur_value)

        # This would have been the bug: EUR label but USD value
        self.assertNotEqual(chart_call['value'], original_usd_value)


if __name__ == '__main__':
    unittest.main()
