"""
Unit tests for FX currency conversion in the UI layer.
Prevents regression of the YTD analytics currency conversion bug.
"""

import unittest
from datetime import date, datetime
from decimal import Decimal

from src.portfolio.models import (
    PortfolioSnapshot, Position, FinancialInstrument, Currency,
    InstrumentType
)
from src.ui.streamlit_app import PortfolioTrackerUI


class MockDataManager:
    """Mock data manager for testing FX rates."""

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


class TestFXCurrencyConversion(unittest.TestCase):
    """Test FX currency conversion functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.ui = PortfolioTrackerUI()
        self.portfolio_manager = MockPortfolioManager()

        # Create test instruments
        self.btc_instrument = FinancialInstrument(
            symbol="BTC-USD",
            name="Bitcoin",
            instrument_type=InstrumentType.CRYPTO,
            currency=Currency.USD,
            isin=None
        )

        self.eth_instrument = FinancialInstrument(
            symbol="ETH-USD",
            name="Ethereum",
            instrument_type=InstrumentType.CRYPTO,
            currency=Currency.USD,
            isin=None
        )

        # Create test positions
        self.btc_position = Position(
            instrument=self.btc_instrument,
            quantity=Decimal("0.5"),
            average_cost=Decimal("95000.00"),
            current_price=Decimal("98000.00"),
            last_updated=datetime.now()
        )

        self.eth_position = Position(
            instrument=self.eth_instrument,
            quantity=Decimal("2.0"),
            average_cost=Decimal("3200.00"),
            current_price=Decimal("3400.00"),
            last_updated=datetime.now()
        )

        # Create test snapshot
        self.test_snapshot = PortfolioSnapshot(
            date=date.today(),
            total_value=Decimal("60000.00"),
            cash_balance=Decimal("5000.00"),
            positions_value=Decimal("55000.00"),
            base_currency=Currency.USD,
            positions={
                "BTC-USD": self.btc_position,
                "ETH-USD": self.eth_position
            },
            cash_balances={Currency.USD: Decimal("5000.00")},
            total_cost_basis=Decimal("58000.00"),
            total_unrealized_pnl=Decimal("2000.00"),
            total_unrealized_pnl_percent=Decimal("3.45")
        )

    def test_convert_snapshots_to_currency_basic(self):
        """Test basic snapshot currency conversion."""
        # Convert USD to EUR
        converted_snapshots = self.ui._convert_snapshots_to_currency(
            [self.test_snapshot], "EUR", self.portfolio_manager
        )

        self.assertEqual(len(converted_snapshots), 1)

        converted_snapshot = converted_snapshots[0]

        # Check currency conversion (USD to EUR rate is 0.85)
        expected_total_value = self.test_snapshot.total_value * Decimal("0.85")
        self.assertEqual(converted_snapshot.total_value, expected_total_value)
        self.assertEqual(converted_snapshot.base_currency, Currency.EUR)

        # Check that positions are preserved and still valid Position objects
        self.assertEqual(len(converted_snapshot.positions), len(self.test_snapshot.positions))
        for key, position in converted_snapshot.positions.items():
            self.assertIsInstance(position, Position)
            self.assertEqual(position.instrument.symbol, self.test_snapshot.positions[key].instrument.symbol)

    def test_convert_snapshots_same_currency(self):
        """Test that same currency snapshots are passed through unchanged."""
        converted_snapshots = self.ui._convert_snapshots_to_currency(
            [self.test_snapshot], "USD", self.portfolio_manager
        )

        self.assertEqual(len(converted_snapshots), 1)
        # Should be the exact same object
        self.assertIs(converted_snapshots[0], self.test_snapshot)

    def test_convert_multiple_snapshots(self):
        """Test conversion of multiple snapshots (YTD scenario)."""
        # Create multiple snapshots
        snapshots = []
        for i in range(3):
            snapshot_date = date(2024, 1 + i, 1)
            snapshot = PortfolioSnapshot(
                date=snapshot_date,
                total_value=Decimal("50000.00") * (1 + i * Decimal("0.1")),  # Growing value
                cash_balance=Decimal("5000.00"),
                positions_value=Decimal("45000.00") * (1 + i * Decimal("0.1")),
                base_currency=Currency.USD,
                positions={"BTC-USD": self.btc_position},
                cash_balances={Currency.USD: Decimal("5000.00")},
                total_cost_basis=Decimal("48000.00"),
                total_unrealized_pnl=Decimal("2000.00") * (1 + i * Decimal("0.1")),
                total_unrealized_pnl_percent=Decimal("4.17")
            )
            snapshots.append(snapshot)

        # Convert all to GBP
        converted_snapshots = self.ui._convert_snapshots_to_currency(
            snapshots, "GBP", self.portfolio_manager
        )

        self.assertEqual(len(converted_snapshots), 3)

        # Check each conversion
        gbp_rate = Decimal("0.80")  # USD to GBP
        for i, (original, converted) in enumerate(zip(snapshots, converted_snapshots)):
            expected_value = original.total_value * gbp_rate
            self.assertEqual(converted.total_value, expected_value)
            self.assertEqual(converted.base_currency, Currency.GBP)
            self.assertEqual(converted.date, original.date)
            self.assertEqual(len(converted.positions), len(original.positions))

    def test_convert_snapshots_with_position_validation_issue(self):
        """Test that Position objects don't cause Pydantic validation errors."""
        # This is the specific issue that was fixed - Position objects were causing
        # validation errors when passed to new PortfolioSnapshot constructor

        # Create a snapshot with complex positions
        positions_dict = {
            "BTC-USD": self.btc_position,
            "ETH-USD": self.eth_position
        }

        snapshot_with_positions = PortfolioSnapshot(
            date=date.today(),
            total_value=Decimal("120000.00"),
            cash_balance=Decimal("5000.00"),
            positions_value=Decimal("115000.00"),
            base_currency=Currency.USD,
            positions=positions_dict,
            cash_balances={Currency.USD: Decimal("5000.00")},
            total_cost_basis=Decimal("118000.00"),
            total_unrealized_pnl=Decimal("2000.00"),
            total_unrealized_pnl_percent=Decimal("1.69")
        )

        # This should not raise a ValidationError
        converted_snapshots = self.ui._convert_snapshots_to_currency(
            [snapshot_with_positions], "EUR", self.portfolio_manager
        )

        self.assertEqual(len(converted_snapshots), 1)
        converted = converted_snapshots[0]

        # Verify positions are preserved correctly
        self.assertEqual(len(converted.positions), 2)
        self.assertIn("BTC-USD", converted.positions)
        self.assertIn("ETH-USD", converted.positions)

        # Verify each position is still a valid Position object
        for key, position in converted.positions.items():
            self.assertIsInstance(position, Position)
            original_position = snapshot_with_positions.positions[key]
            self.assertEqual(position.instrument.symbol, original_position.instrument.symbol)
            self.assertEqual(position.quantity, original_position.quantity)

    def test_convert_empty_snapshots_list(self):
        """Test conversion of empty snapshots list."""
        converted_snapshots = self.ui._convert_snapshots_to_currency(
            [], "EUR", self.portfolio_manager
        )

        self.assertEqual(converted_snapshots, [])

    def test_convert_snapshots_error_handling(self):
        """Test error handling when conversion fails."""
        # Create a failing portfolio manager
        class FailingDataManager:
            def get_exchange_rate(self, from_currency, to_currency):
                raise Exception("API failure")

            def get_historical_fx_rate_on(self, date_val, from_currency, to_currency):
                raise Exception("API failure")

        class FailingPortfolioManager:
            def __init__(self):
                self.data_manager = FailingDataManager()

        failing_manager = FailingPortfolioManager()

        # Should handle error gracefully and return original snapshot
        converted_snapshots = self.ui._convert_snapshots_to_currency(
            [self.test_snapshot], "EUR", failing_manager
        )

        self.assertEqual(len(converted_snapshots), 1)
        # Should return the original snapshot when conversion fails
        self.assertIs(converted_snapshots[0], self.test_snapshot)


if __name__ == '__main__':
    unittest.main()
