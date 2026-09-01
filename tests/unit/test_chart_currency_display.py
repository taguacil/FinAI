"""
Unit tests for chart currency display in the analytics page.
Ensures the portfolio value chart shows data in the selected display currency.

The current analytics chart is ``PortfolioTrackerUI.plot_portfolio_value_chart(
history_df, display_currency_code, portfolio_manager, start_date, end_date, ...)``.
It renders the values contained in ``history_df["total_value"]`` and labels the
axes/title with ``display_currency_code``. Values are converted to the display
currency by the caller via ``_convert_to_base`` before being placed in the
DataFrame, so this suite exercises the convert-then-plot flow.

NOTE: ``plot_portfolio_value_chart`` itself does not convert values; it plots
exactly what it is given. These tests therefore verify that (a) ``_convert_to_base``
produces correct display-currency values and (b) the chart faithfully renders
those values with a matching currency label (no value/label mismatch).
"""

import unittest
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd

import src.ui.streamlit_app as app_mod
from src.portfolio.models import Currency
from src.ui.streamlit_app import PortfolioTrackerUI


class MockFXCache:
    """Mock FX cache exposing the interface used by _convert_to_base."""

    def __init__(self, rates):
        self.rates = rates

    def get_current_rate(self, from_currency, to_currency):
        return self.rates.get((from_currency, to_currency))

    def get_rate(self, from_currency, to_currency, rate_date):
        return self.rates.get((from_currency, to_currency))


class MockDataManager:
    """Mock data manager providing cached FX rates."""

    def __init__(self):
        self.rates = {
            (Currency.USD, Currency.EUR): Decimal("0.85"),
            (Currency.EUR, Currency.USD): Decimal("1.18"),
            (Currency.USD, Currency.GBP): Decimal("0.80"),
            (Currency.GBP, Currency.USD): Decimal("1.25"),
        }
        self.fx_cache = MockFXCache(self.rates)

    def get_exchange_rate(self, from_currency, to_currency):
        return self.rates.get((from_currency, to_currency))

    def get_historical_fx_rate_on(self, rate_date, from_currency, to_currency):
        return self.rates.get((from_currency, to_currency))


class MockPortfolioManager:
    """Mock portfolio manager exposing a ``data_manager`` attribute."""

    def __init__(self):
        self.data_manager = MockDataManager()


class TestChartCurrencyDisplay(unittest.TestCase):
    """Test chart currency display functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui = PortfolioTrackerUI()
        self.portfolio_manager = MockPortfolioManager()

        # USD portfolio values over three months.
        self.dates = [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
        self.usd_values = [Decimal("60000.00"), Decimal("65000.00"), Decimal("70000.00")]

    def _make_history_df(self, values):
        """Build a history DataFrame like get_portfolio_history_filtered returns."""
        idx = pd.to_datetime(self.dates)
        return pd.DataFrame({"total_value": [float(v) for v in values]}, index=idx)

    def _capture_chart(self, history_df, currency_code):
        """Render the chart with streamlit stubbed and return the first plotly figure."""
        captured = {}

        def fake_plotly_chart(fig, **kwargs):
            # First call is the portfolio value figure.
            captured.setdefault("fig", fig)

        with patch.object(app_mod.st, "plotly_chart", side_effect=fake_plotly_chart), \
                patch.object(app_mod.st, "warning"), \
                patch.object(app_mod.st, "subheader"), \
                patch.object(app_mod.st, "caption"):
            self.ui.plot_portfolio_value_chart(
                history_df,
                currency_code,
                self.portfolio_manager,
                date(2024, 1, 1),
                date(2024, 4, 1),
            )
        return captured.get("fig")

    def test_chart_receives_converted_currency_data(self):
        """Chart renders EUR-converted values with an EUR label."""
        eur_values = [
            self.ui._convert_to_base(self.portfolio_manager, v, "USD", "EUR")
            for v in self.usd_values
        ]
        eur_df = self._make_history_df(eur_values)

        fig = self._capture_chart(eur_df, "EUR")
        self.assertIsNotNone(fig)

        # First value should be the EUR-converted value (85% of USD).
        expected_first = float(self.usd_values[0] * Decimal("0.85"))
        self.assertAlmostEqual(list(fig.data[0].y)[0], expected_first, places=6)

        # Axis/title labels reflect the display currency.
        self.assertIn("EUR", fig.layout.title.text)
        self.assertIn("EUR", fig.layout.yaxis.title.text)

    def test_chart_currency_conversion_accuracy(self):
        """Conversion feeding the chart is accurate across currencies."""
        test_currencies = [
            ("EUR", Decimal("0.85")),
            ("GBP", Decimal("0.80")),
            ("USD", Decimal("1.00")),
        ]

        original_value = self.usd_values[0]

        for currency_code, rate in test_currencies:
            with self.subTest(currency=currency_code):
                converted = self.ui._convert_to_base(
                    self.portfolio_manager, original_value, "USD", currency_code
                )
                self.assertEqual(converted, original_value * rate)

    def test_chart_data_consistency_across_snapshots(self):
        """All values are converted consistently, preserving proportions."""
        eur_values = [
            self.ui._convert_to_base(self.portfolio_manager, v, "USD", "EUR")
            for v in self.usd_values
        ]

        # Proportional relationships are preserved after conversion.
        usd_ratio = self.usd_values[1] / self.usd_values[0]
        eur_ratio = eur_values[1] / eur_values[0]
        self.assertAlmostEqual(float(usd_ratio), float(eur_ratio), places=10)

        # And the chart renders each converted value in order.
        eur_df = self._make_history_df(eur_values)
        fig = self._capture_chart(eur_df, "EUR")
        rendered = list(fig.data[0].y)
        for rendered_val, eur_val in zip(rendered, eur_values):
            self.assertAlmostEqual(rendered_val, float(eur_val), places=6)

    def test_chart_fix_prevents_currency_mismatch(self):
        """Chart value and currency label stay consistent (no USD value / EUR label)."""
        original_usd_value = self.usd_values[0]
        eur_values = [
            self.ui._convert_to_base(self.portfolio_manager, v, "USD", "EUR")
            for v in self.usd_values
        ]
        converted_eur_value = eur_values[0]

        # Conversion actually happened and reduced the value (EUR < USD here).
        self.assertNotEqual(original_usd_value, converted_eur_value)
        self.assertLess(converted_eur_value, original_usd_value)

        eur_df = self._make_history_df(eur_values)
        fig = self._capture_chart(eur_df, "EUR")

        # The chart shows the EUR value with an EUR label -- not the USD value.
        first_rendered = list(fig.data[0].y)[0]
        self.assertAlmostEqual(first_rendered, float(converted_eur_value), places=6)
        self.assertNotAlmostEqual(first_rendered, float(original_usd_value), places=6)
        self.assertIn("EUR", fig.layout.title.text)


if __name__ == '__main__':
    unittest.main()
