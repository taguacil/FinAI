"""Tests for Position model fallback functionality."""

import pytest
from decimal import Decimal
from unittest.mock import Mock

from src.portfolio.models import Position, FinancialInstrument, InstrumentType, Currency


class TestPositionFallbacks:
    """Test Position model fallback functionality."""

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
    def position_with_current_price(self, sample_instrument):
        """Create a position with current market price."""
        return Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

    @pytest.fixture
    def position_without_current_price(self, sample_instrument):
        """Create a position without current market price."""
        return Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=None
        )

    @pytest.fixture
    def position_zero_quantity(self, sample_instrument):
        """Create a position with zero quantity."""
        return Position(
            instrument=sample_instrument,
            quantity=Decimal("0"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

    def test_market_value_with_current_price(self, position_with_current_price):
        """Test market value calculation when current price is available."""
        result = position_with_current_price.market_value
        expected = Decimal("100") * Decimal("160.00")
        assert result == expected

    def test_market_value_without_current_price(self, position_without_current_price):
        """Test market value calculation when current price is unavailable."""
        result = position_without_current_price.market_value
        assert result is None

    def test_market_value_zero_quantity(self, position_zero_quantity):
        """Test market value calculation with zero quantity."""
        result = position_zero_quantity.market_value
        assert result == Decimal("0")

    def test_cost_basis_calculation(self, position_with_current_price):
        """Test cost basis calculation."""
        result = position_with_current_price.cost_basis
        expected = Decimal("100") * Decimal("150.00")
        assert result == expected

    def test_cost_basis_zero_quantity(self, position_zero_quantity):
        """Test cost basis calculation with zero quantity."""
        result = position_zero_quantity.cost_basis
        assert result == Decimal("0")

    def test_unrealized_pnl_with_current_price(self, position_with_current_price):
        """Test unrealized P&L calculation when current price is available."""
        result = position_with_current_price.unrealized_pnl
        # Market value: 100 * 160 = 16000, Cost basis: 100 * 150 = 15000
        expected = Decimal("16000") - Decimal("15000")
        assert result == expected

    def test_unrealized_pnl_without_current_price(self, position_without_current_price):
        """Test unrealized P&L calculation when current price is unavailable."""
        result = position_without_current_price.unrealized_pnl
        assert result is None

    def test_unrealized_pnl_zero_quantity(self, position_zero_quantity):
        """Test unrealized P&L calculation with zero quantity."""
        result = position_zero_quantity.unrealized_pnl
        assert result == Decimal("0")

    def test_unrealized_pnl_percent_with_current_price(self, position_with_current_price):
        """Test unrealized P&L percentage calculation when current price is available."""
        result = position_with_current_price.unrealized_pnl_percent
        # P&L: 1000, Cost basis: 15000, Percentage: (1000/15000) * 100 = 6.67%
        expected = (Decimal("1000") / Decimal("15000")) * Decimal("100")
        assert result == expected

    def test_unrealized_pnl_percent_without_current_price(self, position_without_current_price):
        """Test unrealized P&L percentage calculation when current price is unavailable."""
        result = position_without_current_price.unrealized_pnl_percent
        assert result is None

    def test_unrealized_pnl_percent_zero_cost_basis(self, sample_instrument):
        """Test unrealized P&L percentage calculation with zero cost basis."""
        position = Position(
            instrument=sample_instrument,
            quantity=Decimal("0"),
            average_cost=Decimal("0"),
            current_price=Decimal("160.00")
        )
        result = position.unrealized_pnl_percent
        assert result == Decimal("0")

    def test_position_attributes_consistency(self, position_with_current_price):
        """Test that Position object attributes are consistent and accessible."""
        position = position_with_current_price

        # Test all required attributes exist
        assert hasattr(position, 'instrument')
        assert hasattr(position, 'quantity')
        assert hasattr(position, 'average_cost')
        assert hasattr(position, 'current_price')
        assert hasattr(position, 'last_updated')

        # Test attribute types
        assert isinstance(position.instrument, FinancialInstrument)
        assert isinstance(position.quantity, Decimal)
        assert isinstance(position.average_cost, Decimal)
        assert isinstance(position.current_price, Decimal)

        # Test attribute values
        assert position.instrument.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("150.00")
        assert position.current_price == Decimal("160.00")

    def test_position_not_subscriptable(self, position_with_current_price):
        """Test that Position objects are not subscriptable (dict-like)."""
        position = position_with_current_price

        # Should not be subscriptable
        with pytest.raises(TypeError):
            _ = position['quantity']

        with pytest.raises(TypeError):
            _ = position['average_cost']

        with pytest.raises(TypeError):
            _ = position['current_price']

    def test_position_dot_notation_works(self, position_with_current_price):
        """Test that Position object attributes are accessible via dot notation."""
        position = position_with_current_price

        # Should work with dot notation
        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("150.00")
        assert position.current_price == Decimal("160.00")
        assert position.instrument.symbol == "AAPL"

    def test_position_equality(self, sample_instrument):
        """Test Position object equality."""
        position1 = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

        position2 = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

        # Should be equal if all attributes are the same
        assert position1.quantity == position2.quantity
        assert position1.average_cost == position2.average_cost
        assert position1.current_price == position2.current_price
        assert position1.instrument.symbol == position2.instrument.symbol

    def test_position_inequality(self, sample_instrument):
        """Test Position object inequality."""
        position1 = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

        position2 = Position(
            instrument=sample_instrument,
            quantity=Decimal("50"),  # Different quantity
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00")
        )

        # Should be different
        assert position1.quantity != position2.quantity
        assert position1.market_value != position2.market_value

    def test_position_with_foreign_currency(self):
        """Test Position object with foreign currency instrument."""
        foreign_instrument = FinancialInstrument(
            symbol="EURSTOCK",
            name="European Stock",
            isin="DE0001234567",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.EUR
        )

        position = Position(
            instrument=foreign_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00")
        )

        # Should work with foreign currency
        assert position.instrument.currency == Currency.EUR
        assert position.market_value == Decimal("11000.00")  # 100 * 110
        assert position.cost_basis == Decimal("10000.00")   # 100 * 100

    def test_position_edge_cases(self, sample_instrument):
        """Test Position object with edge case values."""
        # Very large numbers
        large_position = Position(
            instrument=sample_instrument,
            quantity=Decimal("999999999"),
            average_cost=Decimal("999999.99"),
            current_price=Decimal("999999.99")
        )

        # Very small numbers
        small_position = Position(
            instrument=sample_instrument,
            quantity=Decimal("0.000001"),
            average_cost=Decimal("0.01"),
            current_price=Decimal("0.01")
        )

        # Test calculations work
        assert large_position.market_value is not None
        assert small_position.market_value is not None
        assert large_position.cost_basis is not None
        assert small_position.cost_basis is not None

    def test_position_serialization_compatibility(self, position_with_current_price):
        """Test that Position objects can be serialized for storage."""
        position = position_with_current_price

        # Test that all attributes can be accessed for serialization
        serializable_data = {
            'instrument': position.instrument,
            'quantity': position.quantity,
            'average_cost': position.average_cost,
            'current_price': position.current_price,
            'last_updated': position.last_updated
        }

        # Should not raise any errors
        assert len(serializable_data) == 5
        assert all(key in serializable_data for key in ['instrument', 'quantity', 'average_cost', 'current_price', 'last_updated'])

    def test_position_methods_return_correct_types(self, position_with_current_price):
        """Test that Position object methods return correct types."""
        position = position_with_current_price

        # Test return types
        assert isinstance(position.market_value, Decimal)
        assert isinstance(position.cost_basis, Decimal)
        assert isinstance(position.unrealized_pnl, Decimal)
        assert isinstance(position.unrealized_pnl_percent, Decimal)

    def test_position_with_none_values(self, sample_instrument):
        """Test Position object behavior with None values."""
        position = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=None,  # No current price
            last_updated=None    # No last updated
        )

        # Should handle None values gracefully
        assert position.current_price is None
        assert position.last_updated is None
        assert position.market_value is None
        assert position.unrealized_pnl is None
        assert position.unrealized_pnl_percent is None

        # Cost basis should still work
        assert position.cost_basis == Decimal("15000.00")
