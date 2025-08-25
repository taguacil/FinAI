"""Tests for Portfolio model fallback functionality."""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch

from src.portfolio.models import (
    Portfolio, Position, FinancialInstrument, InstrumentType,
    Transaction, TransactionType, Currency
)


class TestPortfolioFallbacks:
    """Test Portfolio model fallback functionality."""

    @pytest.fixture
    def sample_instrument(self):
        """Create a sample financial instrument."""
        return FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            isin="US0378331005",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio."""
        return Portfolio(
            id="test-portfolio-123",
            name="Test Portfolio",
            base_currency=Currency.USD
        )

    @pytest.fixture
    def sample_position_with_current_price(self, sample_instrument):
        """Create a position with current market price."""
        return Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

    @pytest.fixture
    def sample_position_without_current_price(self, sample_instrument):
        """Create a position without current market price."""
        return Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=None
        )

    def test_get_total_value_with_rate_function_all_usd(self, sample_portfolio, sample_position_with_current_price):
        """Test portfolio value calculation when all positions are in base currency."""
        # Setup
        sample_portfolio.positions = {"AAPL": sample_position_with_current_price}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}

        # Mock rate function
        def mock_rate_function(from_currency, to_currency):
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Expected: Cash (1000) + Position (100 * 160 = 16000)
        expected = Decimal("1000.00") + Decimal("16000.00")
        assert result == expected

    def test_get_total_value_with_rate_function_foreign_currency(self, sample_portfolio, sample_instrument):
        """Test portfolio value calculation with foreign currency positions."""
        # Setup
        foreign_position = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00")
        )
        # Change instrument currency to EUR
        foreign_position.instrument.currency = Currency.EUR

        sample_portfolio.positions = {"EURSTOCK": foreign_position}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}

        # Mock rate function to return EUR/USD rate
        def mock_rate_function(from_currency, to_currency):
            if from_currency == Currency.EUR and to_currency == Currency.USD:
                return Decimal("1.2")
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Expected: Cash (1000) + Position (100 * 110 * 1.2 = 13200)
        expected = Decimal("1000.00") + Decimal("13200.00")
        assert result == expected

    def test_get_total_value_with_rate_function_no_exchange_rate(self, sample_portfolio, sample_instrument):
        """Test portfolio value calculation when exchange rate is unavailable."""
        # Setup
        foreign_position = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00")
        )
        # Change instrument currency to EUR
        foreign_position.instrument.currency = Currency.EUR

        sample_portfolio.positions = {"EURSTOCK": foreign_position}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}

        # Mock rate function to return None for foreign currency
        def mock_rate_function(from_currency, to_currency):
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Should only include USD cash since EUR position has no exchange rate
        expected = Decimal("1000.00")
        assert result == expected

    def test_get_total_value_with_rate_function_mixed_currencies(self, sample_portfolio, sample_instrument):
        """Test portfolio value calculation with mixed currency positions."""
        # Setup USD position
        usd_position = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

        # Setup EUR position
        eur_instrument = FinancialInstrument(
            symbol="EURSTOCK",
            name="European Stock",
            isin="DE0001234567",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.EUR
        )
        eur_position = Position(
            instrument=eur_instrument,
            quantity=Decimal("50"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00")
        )

        sample_portfolio.positions = {"AAPL": usd_position, "EURSTOCK": eur_position}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00"), Currency.EUR: Decimal("500.00")}

        # Mock rate function
        def mock_rate_function(from_currency, to_currency):
            if from_currency == Currency.EUR and to_currency == Currency.USD:
                return Decimal("1.2")
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Expected: USD Cash (1000) + EUR Cash (500 * 1.2 = 600) + USD Position (100 * 160 = 16000) + EUR Position (50 * 110 * 1.2 = 6600)
        expected = Decimal("1000.00") + Decimal("600.00") + Decimal("16000.00") + Decimal("6600.00")
        assert result == expected

    def test_get_total_value_with_rate_function_no_positions(self, sample_portfolio):
        """Test portfolio value calculation with no positions."""
        # Setup
        sample_portfolio.positions = {}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}

        def mock_rate_function(from_currency, to_currency):
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Should only include cash
        assert result == Decimal("1000.00")

    def test_get_total_value_with_rate_function_no_cash(self, sample_portfolio, sample_position_with_current_price):
        """Test portfolio value calculation with no cash balances."""
        # Setup
        sample_portfolio.positions = {"AAPL": sample_position_with_current_price}
        sample_portfolio.cash_balances = {}

        def mock_rate_function(from_currency, to_currency):
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Should only include position value
        expected = Decimal("100") * Decimal("160.00")
        assert result == expected

    def test_get_total_value_with_rate_function_position_without_current_price(self, sample_portfolio, sample_position_without_current_price):
        """Test portfolio value calculation when position has no current price."""
        # Setup
        sample_portfolio.positions = {"AAPL": sample_position_without_current_price}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}

        def mock_rate_function(from_currency, to_currency):
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Should only include cash since position has no current price
        assert result == Decimal("1000.00")

    def test_get_total_value_with_rate_function_zero_quantity_position(self, sample_portfolio, sample_instrument):
        """Test portfolio value calculation with zero quantity position."""
        # Setup
        zero_position = Position(
            instrument=sample_instrument,
            quantity=Decimal("0"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

        sample_portfolio.positions = {"AAPL": zero_position}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}

        def mock_rate_function(from_currency, to_currency):
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Should only include cash since position has zero quantity
        assert result == Decimal("1000.00")

    def test_get_total_value_with_rate_function_none_rate_function(self, sample_portfolio, sample_position_with_current_price):
        """Test portfolio value calculation with None rate function."""
        # Setup
        sample_portfolio.positions = {"AAPL": sample_position_with_current_price}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}

        result = sample_portfolio.get_total_value_with_rate_function(None)

        # Should handle None gracefully and only include USD positions/cash
        expected = Decimal("1000.00") + Decimal("16000.00")
        assert result == expected

    def test_get_total_value_with_rate_function_complex_scenario(self, sample_portfolio):
        """Test portfolio value calculation with complex scenario."""
        # Setup multiple instruments with different currencies
        usd_instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            isin="US0378331005",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD
        )

        eur_instrument = FinancialInstrument(
            symbol="EURSTOCK",
            name="European Stock",
            isin="DE0001234567",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.EUR
        )

        gbp_instrument = FinancialInstrument(
            symbol="GBPSTOCK",
            name="British Stock",
            isin="GB0001234567",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.GBP
        )

        # Create positions
        usd_position = Position(
            instrument=usd_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

        eur_position = Position(
            instrument=eur_instrument,
            quantity=Decimal("50"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00")
        )

        gbp_position = Position(
            instrument=gbp_instrument,
            quantity=Decimal("200"),
            average_cost=Decimal("75.00"),
            current_price=Decimal("80.00")
        )

        sample_portfolio.positions = {
            "AAPL": usd_position,
            "EURSTOCK": eur_position,
            "GBPSTOCK": gbp_position
        }

        sample_portfolio.cash_balances = {
            Currency.USD: Decimal("1000.00"),
            Currency.EUR: Decimal("500.00"),
            Currency.GBP: Decimal("300.00")
        }

        # Mock rate function with realistic rates
        def mock_rate_function(from_currency, to_currency):
            if from_currency == to_currency:
                return Decimal("1.0")
            elif from_currency == Currency.EUR and to_currency == Currency.USD:
                return Decimal("1.2")
            elif from_currency == Currency.GBP and to_currency == Currency.USD:
                return Decimal("1.3")
            return None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Calculate expected values
        usd_cash = Decimal("1000.00")
        eur_cash_usd = Decimal("500.00") * Decimal("1.2")  # 600
        gbp_cash_usd = Decimal("300.00") * Decimal("1.3")  # 390

        usd_position_value = Decimal("100") * Decimal("160.00")  # 16000
        eur_position_value_usd = Decimal("50") * Decimal("110.00") * Decimal("1.2")  # 6600
        gbp_position_value_usd = Decimal("200") * Decimal("80.00") * Decimal("1.3")  # 20800

        expected = usd_cash + eur_cash_usd + gbp_cash_usd + usd_position_value + eur_position_value_usd + gbp_position_value_usd

        assert result == expected

    def test_get_total_value_with_rate_function_error_handling(self, sample_portfolio, sample_position_with_current_price):
        """Test portfolio value calculation error handling."""
        # Setup
        sample_portfolio.positions = {"AAPL": sample_position_with_current_price}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}

        # Mock rate function that raises an error
        def mock_rate_function(from_currency, to_currency):
            if from_currency == Currency.USD and to_currency == Currency.USD:
                return Decimal("1.0")
            raise Exception("Rate function error")

        # Should handle errors gracefully
        with pytest.raises(Exception, match="Rate function error"):
            sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

    def test_get_total_value_with_rate_function_decimal_precision(self, sample_portfolio, sample_instrument):
        """Test portfolio value calculation with high decimal precision."""
        # Setup
        precise_position = Position(
            instrument=sample_instrument,
            quantity=Decimal("100.123456"),
            average_cost=Decimal("150.123456"),
            current_price=Decimal("160.123456")
        )

        sample_portfolio.positions = {"AAPL": precise_position}
        sample_portfolio.cash_balances = {Currency.USD: Decimal("1000.123456")}

        def mock_rate_function(from_currency, to_currency):
            return Decimal("1.0") if from_currency == to_currency else None

        result = sample_portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Should maintain precision
        expected_cash = Decimal("1000.123456")
        expected_position = Decimal("100.123456") * Decimal("160.123456")
        expected = expected_cash + expected_position

        assert result == expected
        # Verify high precision is maintained
        assert result.as_tuple()[2] >= 6  # At least 6 decimal places
