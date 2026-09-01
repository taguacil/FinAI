"""
Unit tests for FX currency conversion in the UI layer.
Prevents regression of the analytics currency conversion bug.

The UI converts monetary amounts to a target currency via
``PortfolioTrackerUI._convert_to_base(portfolio_manager, amount,
from_currency_code, base_currency_code, allow_fetch=False)``.

Behavior under test (matches current source in src/ui/streamlit_app.py):
* Same currency (or empty from-code) -> amount returned unchanged.
* allow_fetch=False -> only cached rates are used (fx_cache); if no cached
  rate is available the amount is returned unconverted (no network calls).
* allow_fetch=True -> live rate (get_exchange_rate) with a fallback to
  historical FX (get_historical_fx_rate_on).
* None amount -> Decimal("0").
* Any provider error -> amount returned unconverted.
"""

import unittest
from datetime import date
from decimal import Decimal

from src.portfolio.models import Currency
from src.ui.streamlit_app import PortfolioTrackerUI


class MockFXCache:
    """Mock FX cache exposing the interface used by _convert_to_base."""

    def __init__(self, rates):
        # rates: dict keyed by (Currency, Currency) -> Decimal
        self.rates = rates

    def get_current_rate(self, from_currency, to_currency):
        return self.rates.get((from_currency, to_currency))

    def get_rate(self, from_currency, to_currency, rate_date):
        return self.rates.get((from_currency, to_currency))


class MockDataManager:
    """Mock data manager providing cached + live FX rates."""

    def __init__(self, rates=None):
        if rates is None:
            rates = {
                (Currency.USD, Currency.EUR): Decimal("0.85"),
                (Currency.EUR, Currency.USD): Decimal("1.18"),
                (Currency.USD, Currency.GBP): Decimal("0.80"),
                (Currency.GBP, Currency.USD): Decimal("1.25"),
            }
        self.rates = rates
        self.fx_cache = MockFXCache(rates)

    def get_exchange_rate(self, from_currency, to_currency):
        return self.rates.get((from_currency, to_currency))

    def get_historical_fx_rate_on(self, rate_date, from_currency, to_currency):
        return self.rates.get((from_currency, to_currency))


class MockPortfolioManager:
    """Mock portfolio manager exposing a ``data_manager`` attribute."""

    def __init__(self, rates=None):
        self.data_manager = MockDataManager(rates)


class TestFXCurrencyConversion(unittest.TestCase):
    """Test FX currency conversion functionality in the UI layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui = PortfolioTrackerUI()
        self.portfolio_manager = MockPortfolioManager()

    def test_convert_amount_to_currency_basic(self):
        """USD amount converts to EUR using the cached rate (0.85)."""
        amount = Decimal("60000.00")

        converted = self.ui._convert_to_base(
            self.portfolio_manager, amount, "USD", "EUR"
        )

        self.assertEqual(converted, amount * Decimal("0.85"))
        self.assertIsInstance(converted, Decimal)

    def test_convert_amount_same_currency(self):
        """Same-currency conversion returns the amount unchanged."""
        amount = Decimal("60000.00")

        converted = self.ui._convert_to_base(
            self.portfolio_manager, amount, "USD", "USD"
        )

        self.assertEqual(converted, amount)

        # An empty/None from-currency code is also treated as a no-op.
        converted_empty = self.ui._convert_to_base(
            self.portfolio_manager, amount, "", "EUR"
        )
        self.assertEqual(converted_empty, amount)

    def test_convert_multiple_currencies(self):
        """Conversion is correct across several target currencies."""
        amount = Decimal("50000.00")

        expected = {
            "EUR": amount * Decimal("0.85"),
            "GBP": amount * Decimal("0.80"),
            "USD": amount,  # passthrough
        }

        for currency_code, expected_value in expected.items():
            with self.subTest(currency=currency_code):
                converted = self.ui._convert_to_base(
                    self.portfolio_manager, amount, "USD", currency_code
                )
                self.assertEqual(converted, expected_value)

    def test_convert_uses_live_rate_when_fetch_allowed(self):
        """allow_fetch=True uses the live exchange rate, with historical fallback."""
        amount = Decimal("120000.00")

        # Live rate available.
        converted = self.ui._convert_to_base(
            self.portfolio_manager, amount, "USD", "EUR", allow_fetch=True
        )
        self.assertEqual(converted, amount * Decimal("0.85"))

        # Live rate missing -> falls back to historical FX rate.
        class LiveMissingDataManager(MockDataManager):
            def get_exchange_rate(self, from_currency, to_currency):
                return None

        pm = MockPortfolioManager()
        pm.data_manager = LiveMissingDataManager()
        converted_hist = self.ui._convert_to_base(
            pm, amount, "USD", "GBP", allow_fetch=True
        )
        self.assertEqual(converted_hist, amount * Decimal("0.80"))

    def test_convert_none_and_missing_rate(self):
        """None amount yields Decimal(0); missing cached rate returns amount unconverted."""
        # None amount -> Decimal("0")
        self.assertEqual(
            self.ui._convert_to_base(self.portfolio_manager, None, "USD", "EUR"),
            Decimal("0"),
        )

        # No cached rate available (empty rate table) -> returned unconverted,
        # with no network fetch attempted.
        empty_pm = MockPortfolioManager(rates={})
        amount = Decimal("60000.00")
        converted = self.ui._convert_to_base(empty_pm, amount, "USD", "EUR")
        self.assertEqual(converted, amount)

    def test_convert_error_handling(self):
        """Provider errors are swallowed and the amount is returned unconverted."""

        class FailingFXCache:
            def get_current_rate(self, from_currency, to_currency):
                raise Exception("cache failure")

            def get_rate(self, from_currency, to_currency, rate_date):
                raise Exception("cache failure")

        class FailingDataManager:
            def __init__(self):
                self.fx_cache = FailingFXCache()

            def get_exchange_rate(self, from_currency, to_currency):
                raise Exception("API failure")

            def get_historical_fx_rate_on(self, rate_date, from_currency, to_currency):
                raise Exception("API failure")

        class FailingPortfolioManager:
            def __init__(self):
                self.data_manager = FailingDataManager()

        failing_manager = FailingPortfolioManager()
        amount = Decimal("60000.00")

        # allow_fetch=False path (cache raises) -> unconverted amount returned.
        converted = self.ui._convert_to_base(failing_manager, amount, "USD", "EUR")
        self.assertEqual(converted, amount)

        # allow_fetch=True path (live + historical raise) -> unconverted too.
        converted_fetch = self.ui._convert_to_base(
            failing_manager, amount, "USD", "EUR", allow_fetch=True
        )
        self.assertEqual(converted_fetch, amount)


if __name__ == '__main__':
    unittest.main()
