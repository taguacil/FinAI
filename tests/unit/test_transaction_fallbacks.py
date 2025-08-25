"""Tests for transaction-based fallback functionality."""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.portfolio.manager import PortfolioManager
from src.portfolio.models import (
    Portfolio, Position, FinancialInstrument, InstrumentType,
    Transaction, TransactionType, Currency, PortfolioSnapshot
)
from src.portfolio.storage import FileBasedStorage
from src.data_providers.manager import DataProviderManager


class TestTransactionFallbacks:
    """Test transaction-based fallback functionality."""

    @pytest.fixture
    def portfolio_manager(self):
        """Create a portfolio manager with mocked dependencies."""
        storage = Mock(spec=FileBasedStorage)
        data_manager = Mock(spec=DataProviderManager)
        return PortfolioManager(storage=storage, data_manager=data_manager)

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        return Portfolio(
            id="test-portfolio-123",
            name="Test Portfolio",
            base_currency=Currency.USD,
            created_at=datetime.now() - timedelta(days=10)
        )

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
    def sample_transaction(self, sample_instrument):
        """Create a sample transaction."""
        return Transaction(
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now() - timedelta(days=5),
            currency=Currency.USD,
            notes="Test purchase"
        )

    def test_get_portfolio_value_with_fallbacks_enabled(self, portfolio_manager, sample_portfolio):
        """Test portfolio value calculation with fallbacks enabled."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Mock the rate function to return None (simulating unavailable rates)
        with patch.object(portfolio_manager, '_get_exchange_rate', return_value=None):
            with patch.object(portfolio_manager, '_get_historical_exchange_rate_fallback', return_value=Decimal("1.0")):
                # Mock portfolio's get_total_value_with_rate_function
                mock_portfolio = Mock()
                mock_portfolio.get_total_value_with_rate_function.return_value = Decimal("15000.00")
                portfolio_manager.current_portfolio = mock_portfolio

                result = portfolio_manager.get_portfolio_value(use_transaction_fallbacks=True)

                assert result == Decimal("15000.00")
                mock_portfolio.get_total_value_with_rate_function.assert_called_once()

    def test_get_portfolio_value_without_fallbacks(self, portfolio_manager, sample_portfolio):
        """Test portfolio value calculation without fallbacks."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Mock the rate function to return None
        with patch.object(portfolio_manager, '_get_exchange_rate', return_value=None):
            # Mock portfolio's get_total_value_with_rate_function
            mock_portfolio = Mock()
            mock_portfolio.get_total_value_with_rate_function.return_value = Decimal("0.00")
            portfolio_manager.current_portfolio = mock_portfolio

            result = portfolio_manager.get_portfolio_value(use_transaction_fallbacks=False)

            assert result == Decimal("0.00")

    def test_get_portfolio_value_no_portfolio(self, portfolio_manager):
        """Test portfolio value calculation when no portfolio is loaded."""
        result = portfolio_manager.get_portfolio_value()
        assert result == Decimal("0")

    def test_historical_exchange_rate_fallback_same_currency(self, portfolio_manager):
        """Test historical exchange rate fallback for same currency."""
        result = portfolio_manager._get_historical_exchange_rate_fallback(
            Currency.USD, Currency.USD
        )
        assert result == Decimal("1")

    def test_historical_exchange_rate_fallback_no_portfolio(self, portfolio_manager):
        """Test historical exchange rate fallback when no portfolio is loaded."""
        result = portfolio_manager._get_historical_exchange_rate_fallback(
            Currency.USD, Currency.EUR
        )
        assert result is None

    def test_historical_exchange_rate_fallback_with_transactions(self, portfolio_manager, sample_portfolio, sample_transaction):
        """Test historical exchange rate fallback using transaction history."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio
        sample_portfolio.transactions = [sample_transaction]

        # Mock data manager to return a historical rate
        with patch.object(portfolio_manager.data_manager, 'get_historical_fx_rate_on', return_value=Decimal("1.2")):
            result = portfolio_manager._get_historical_exchange_rate_fallback(
                Currency.USD, Currency.EUR
            )
            assert result == Decimal("1.2")

    def test_create_prefilled_snapshots_from_transaction(self, portfolio_manager, sample_portfolio):
        """Test creation of prefilled snapshots from transaction date."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio
        transaction_date = date.today() - timedelta(days=3)

        # Mock the snapshot creation method
        with patch.object(portfolio_manager, 'create_snapshot_for_date_with_fallbacks') as mock_create:
            mock_snapshot = Mock(spec=PortfolioSnapshot)
            mock_create.return_value = mock_snapshot

            result = portfolio_manager.create_prefilled_snapshots_from_transaction(transaction_date)

            # Should create snapshots for 4 days (transaction date + 3 days to today)
            assert len(result) == 4
            assert mock_create.call_count == 4

    def test_create_prefilled_snapshots_future_date(self, portfolio_manager, sample_portfolio):
        """Test prefilled snapshots creation with future transaction date."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio
        future_date = date.today() + timedelta(days=5)

        result = portfolio_manager.create_prefilled_snapshots_from_transaction(future_date)

        assert result == []

    def test_create_prefilled_snapshots_no_portfolio(self, portfolio_manager):
        """Test prefilled snapshots creation when no portfolio is loaded."""
        with pytest.raises(ValueError, match="No portfolio loaded"):
            portfolio_manager.create_prefilled_snapshots_from_transaction(date.today())

    def test_create_snapshot_for_date_with_fallbacks(self, portfolio_manager, sample_portfolio):
        """Test snapshot creation for a specific date with fallbacks."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Mock the reconstruction method
        mock_state = {
            'positions': {},
            'cash_balances': {Currency.USD: Decimal("1000.00")}
        }
        with patch.object(portfolio_manager, '_reconstruct_portfolio_state_as_of', return_value=mock_state):
            with patch.object(portfolio_manager, '_calculate_portfolio_value_as_of', return_value=Decimal("1000.00")):
                result = portfolio_manager.create_snapshot_for_date_with_fallbacks(date.today())

                assert isinstance(result, PortfolioSnapshot)
                assert result.total_value == Decimal("1000.00")
                assert result.cash_balance == Decimal("1000.00")

    def test_reconstruct_portfolio_state_as_of(self, portfolio_manager, sample_portfolio, sample_transaction):
        """Test portfolio state reconstruction for a specific date."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio
        sample_portfolio.transactions = [sample_transaction]
        target_date = sample_transaction.timestamp.date()

        result = portfolio_manager._reconstruct_portfolio_state_as_of(target_date)

        assert 'positions' in result
        assert 'cash_balances' in result
        assert Currency.USD in result['cash_balances']

        # Check that positions are Position objects, not dictionaries
        for position in result['positions'].values():
            assert isinstance(position, Position)
            assert hasattr(position, 'quantity')
            assert hasattr(position, 'average_cost')
            assert hasattr(position, 'current_price')

    def test_reconstruct_portfolio_state_buy_transaction(self, portfolio_manager, sample_portfolio, sample_instrument):
        """Test portfolio state reconstruction with BUY transaction."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Create a BUY transaction
        buy_transaction = Transaction(
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now() - timedelta(days=1),
            currency=Currency.USD,
            notes="Test buy"
        )
        sample_portfolio.transactions = [buy_transaction]

        result = portfolio_manager._reconstruct_portfolio_state_as_of(date.today())

        # Should have one position
        assert len(result['positions']) == 1
        position = result['positions']['AAPL']
        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("150.00")
        assert position.current_price == Decimal("150.00")

    def test_reconstruct_portfolio_state_sell_transaction(self, portfolio_manager, sample_portfolio, sample_instrument):
        """Test portfolio state reconstruction with SELL transaction."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Create BUY then SELL transactions
        buy_transaction = Transaction(
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now() - timedelta(days=2),
            currency=Currency.USD,
            notes="Test buy"
        )

        sell_transaction = Transaction(
            instrument=sample_instrument,
            transaction_type=TransactionType.SELL,
            quantity=Decimal("50"),
            price=Decimal("160.00"),
            timestamp=datetime.now() - timedelta(days=1),
            currency=Currency.USD,
            notes="Test sell"
        )

        sample_portfolio.transactions = [buy_transaction, sell_transaction]

        result = portfolio_manager._reconstruct_portfolio_state_as_of(date.today())

        # Should have one position with reduced quantity
        assert len(result['positions']) == 1
        position = result['positions']['AAPL']
        assert position.quantity == Decimal("50")
        assert position.average_cost == Decimal("150.00")  # Average cost doesn't change on sell

    def test_calculate_portfolio_value_as_of(self, portfolio_manager, sample_portfolio):
        """Test portfolio value calculation for a specific date."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Create mock portfolio state
        mock_position = Mock(spec=Position)
        mock_position.quantity = Decimal("100")
        mock_position.current_price = Decimal("150.00")
        mock_position.instrument.currency = Currency.USD

        mock_state = {
            'positions': {'AAPL': mock_position},
            'cash_balances': {Currency.USD: Decimal("1000.00")}
        }

        result = portfolio_manager._calculate_portfolio_value_as_of(date.today(), mock_state)

        # Should be cash (1000) + position value (100 * 150 = 15000)
        expected = Decimal("1000.00") + Decimal("15000.00")
        assert result == expected

    def test_calculate_portfolio_value_foreign_currency(self, portfolio_manager, sample_portfolio):
        """Test portfolio value calculation with foreign currency positions."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Create mock portfolio state with EUR position
        mock_position = Mock(spec=Position)
        mock_position.quantity = Decimal("100")
        mock_position.current_price = Decimal("100.00")
        mock_position.instrument.currency = Currency.EUR

        mock_state = {
            'positions': {'EURSTOCK': mock_position},
            'cash_balances': {Currency.USD: Decimal("1000.00")}
        }

        # Mock exchange rate
        with patch.object(portfolio_manager, '_get_historical_exchange_rate_for_date', return_value=Decimal("1.2")):
            result = portfolio_manager._calculate_portfolio_value_as_of(date.today(), mock_state)

            # Should be cash (1000) + position value (100 * 100 * 1.2 = 12000)
            expected = Decimal("1000.00") + Decimal("12000.00")
            assert result == expected

    def test_get_historical_exchange_rate_for_date_same_currency(self, portfolio_manager):
        """Test historical exchange rate for same currency."""
        result = portfolio_manager._get_historical_exchange_rate_for_date(
            date.today(), Currency.USD, Currency.USD
        )
        assert result == Decimal("1")

    def test_get_historical_exchange_rate_for_date_success(self, portfolio_manager):
        """Test successful historical exchange rate retrieval."""
        # Mock data manager to return a rate
        with patch.object(portfolio_manager.data_manager, 'get_historical_fx_rate_on', return_value=Decimal("1.2")):
            result = portfolio_manager._get_historical_exchange_rate_for_date(
                date.today(), Currency.USD, Currency.EUR
            )
            assert result == Decimal("1.2")

    def test_get_historical_exchange_rate_for_date_fallback_search(self, portfolio_manager):
        """Test historical exchange rate fallback search within time window."""
        # Mock data manager to fail on exact date but succeed on nearby date
        def mock_get_rate(target_date, from_curr, to_curr):
            if target_date == date.today():
                return None
            elif target_date == date.today() + timedelta(days=1):
                return Decimal("1.2")
            return None

        with patch.object(portfolio_manager.data_manager, 'get_historical_fx_rate_on', side_effect=mock_get_rate):
            result = portfolio_manager._get_historical_exchange_rate_for_date(
                date.today(), Currency.USD, Currency.EUR
            )
            assert result == Decimal("1.2")

    def test_refresh_snapshots_with_current_data(self, portfolio_manager, sample_portfolio):
        """Test manual snapshot refresh functionality."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Mock storage to return existing snapshots
        mock_snapshot = Mock(spec=PortfolioSnapshot)
        mock_snapshot.date = date.today()

        with patch.object(portfolio_manager.storage, 'get_snapshots_in_range', return_value=[mock_snapshot]):
            with patch.object(portfolio_manager, '_update_snapshot_with_current_data', return_value=mock_snapshot):
                result = portfolio_manager.refresh_snapshots_with_current_data()

                assert len(result) == 1
                assert result[0] == mock_snapshot

    def test_refresh_snapshots_no_portfolio(self, portfolio_manager):
        """Test snapshot refresh when no portfolio is loaded."""
        with pytest.raises(ValueError, match="No portfolio loaded"):
            portfolio_manager.refresh_snapshots_with_current_data()

    def test_update_snapshot_with_current_data(self, portfolio_manager, sample_portfolio):
        """Test updating existing snapshot with current market data."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Create mock snapshot
        mock_snapshot = Mock(spec=PortfolioSnapshot)
        mock_snapshot.date = date.today()

        # Mock portfolio state reconstruction
        mock_position = Mock(spec=Position)
        mock_position.quantity = Decimal("100")
        mock_position.average_cost = Decimal("150.00")
        mock_position.current_price = Decimal("160.00")
        mock_position.cost_basis = Decimal("15000.00")
        mock_position.unrealized_pnl = Decimal("1000.00")

        mock_state = {
            'positions': {'AAPL': mock_position},
            'cash_balances': {Currency.USD: Decimal("1000.00")}
        }

        with patch.object(portfolio_manager, '_reconstruct_portfolio_state_as_of', return_value=mock_state):
            with patch.object(portfolio_manager, '_calculate_portfolio_value_as_of', return_value=Decimal("16000.00")):
                with patch.object(portfolio_manager.storage, 'save_snapshot'):
                    result = portfolio_manager._update_snapshot_with_current_data(mock_snapshot)

                    assert result == mock_snapshot
                    assert mock_snapshot.total_value == Decimal("16000.00")

    def test_position_objects_are_properly_created(self, portfolio_manager, sample_portfolio, sample_instrument):
        """Test that Position objects are created correctly, not as dictionaries."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Create a transaction
        transaction = Transaction(
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now() - timedelta(days=1),
            currency=Currency.USD,
            notes="Test"
        )
        sample_portfolio.transactions = [transaction]

        # Reconstruct portfolio state
        result = portfolio_manager._reconstruct_portfolio_state_as_of(date.today())

        # Verify positions are Position objects, not dictionaries
        assert len(result['positions']) == 1
        position = result['positions']['AAPL']

        # Should be a Position object, not a dict
        assert isinstance(position, Position)
        assert not isinstance(position, dict)

        # Should have proper attributes
        assert hasattr(position, 'quantity')
        assert hasattr(position, 'average_cost')
        assert hasattr(position, 'current_price')
        assert hasattr(position, 'instrument')

        # Should not be subscriptable (dict-like)
        with pytest.raises(TypeError):
            _ = position['quantity']

        # Should work with dot notation
        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("150.00")
        assert position.current_price == Decimal("150.00")

    def test_position_filtering_zero_quantity(self, portfolio_manager, sample_portfolio, sample_instrument):
        """Test that positions with zero quantity are filtered out."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Create transactions that result in zero quantity
        buy_transaction = Transaction(
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=datetime.now() - timedelta(days=2),
            currency=Currency.USD,
            notes="Buy 100"
        )

        sell_transaction = Transaction(
            instrument=sample_instrument,
            transaction_type=TransactionType.SELL,
            quantity=Decimal("100"),
            price=Decimal("160.00"),
            timestamp=datetime.now() - timedelta(days=1),
            currency=Currency.USD,
            notes="Sell 100"
        )

        sample_portfolio.transactions = [buy_transaction, sell_transaction]

        # Reconstruct portfolio state
        result = portfolio_manager._reconstruct_portfolio_state_as_of(date.today())

        # Should have no positions (all sold)
        assert len(result['positions']) == 0

    def test_error_handling_in_snapshot_creation(self, portfolio_manager, sample_portfolio):
        """Test error handling during snapshot creation."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio

        # Mock reconstruction to raise an error
        with patch.object(portfolio_manager, '_reconstruct_portfolio_state_as_of', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                portfolio_manager.create_snapshot_for_date_with_fallbacks(date.today())

    def test_logging_in_snapshot_creation(self, portfolio_manager, sample_portfolio, caplog):
        """Test that appropriate logging occurs during snapshot creation."""
        # Setup
        portfolio_manager.current_portfolio = sample_portfolio
        transaction_date = date.today() - timedelta(days=3)

        # Mock snapshot creation to fail
        with patch.object(portfolio_manager, 'create_snapshot_for_date_with_fallbacks', side_effect=Exception("Test error")):
            with caplog.at_level('WARNING'):
                result = portfolio_manager.create_prefilled_snapshots_from_transaction(transaction_date)

                # Should log warnings for failed snapshots
                assert "Failed to create snapshot" in caplog.text
                assert "Test error" in caplog.text

                # Should return empty list due to errors
                assert result == []
