"""Tests for PortfolioAnalyzer functionality."""

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.data_providers.manager import DataProviderManager
from src.portfolio.analyzer import PortfolioAnalyzer
from src.portfolio.models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    Portfolio,
    Position,
    Transaction,
    TransactionType,
)
from src.portfolio.storage import FileBasedStorage


class TestPortfolioAnalyzer:
    """Test PortfolioAnalyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_storage = Mock(spec=FileBasedStorage)
        self.mock_data_manager = Mock(spec=DataProviderManager)
        self.analyzer = PortfolioAnalyzer(self.mock_data_manager, self.mock_storage)

    def test_create_snapshot(self):
        """Test creating a single snapshot."""
        # Setup portfolio
        portfolio = Portfolio(
            id="test-id",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Add a position
        instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )
        position = Position(
            instrument=instrument,
            quantity=Decimal("10"),
            average_cost=Decimal("150"),
            current_price=Decimal("160"),
            last_updated=datetime.now(),
        )
        portfolio.positions["AAPL"] = position

        # Mock exchange rate (same currency)
        self.mock_data_manager.get_exchange_rate.return_value = Decimal("1.0")
        # Mock current price fetch (since creating snapshot for today fetches fresh prices)
        self.mock_data_manager.get_current_price.return_value = Decimal("160")

        snapshot = self.analyzer.create_snapshot(portfolio, date.today())

        assert snapshot.total_value == Decimal("1000") + (Decimal("10") * Decimal("160"))
        assert snapshot.cash_balance == Decimal("1000")
        assert snapshot.positions_value == Decimal("1600")
        assert len(snapshot.positions) == 1
        assert snapshot.positions["AAPL"].quantity == Decimal("10")

        # Verify storage save
        self.mock_storage.save_snapshot.assert_called_once()

    def test_create_snapshots_for_range(self):
        """Test creating snapshots for a range."""
        # Setup portfolio with transactions
        portfolio = Portfolio(
            id="test-id",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("10000")},
        )

        instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        # Add transaction
        txn = Transaction(
            id="tx1",
            instrument=instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("10"),
            price=Decimal("150"),
            timestamp=datetime.now() - timedelta(days=5),
            currency=Currency.USD,
        )
        portfolio.add_transaction(txn)

        start_date = date.today() - timedelta(days=5)
        end_date = date.today()

        # Mock historical prices
        # We need to mock _build_historical_price_map or get_historical_prices
        # Since _build_historical_price_map calls get_historical_prices, we mock the latter

        # Mock get_historical_prices to return empty list or some data
        self.mock_data_manager.get_historical_prices.return_value = []
        self.mock_data_manager.get_current_price.return_value = Decimal("160")

        snapshots = self.analyzer.create_snapshots_for_range(portfolio, start_date, end_date)

        assert len(snapshots) == 6  # 5 days ago to today inclusive
        assert snapshots[0].date == start_date
        assert snapshots[-1].date == end_date

        # Verify batch save
        self.mock_storage.save_snapshots_batch.assert_called_once()

    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation."""
        portfolio = Portfolio(
            id="test-id",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000"), Currency.EUR: Decimal("500")},
        )

        # Mock exchange rate for EUR -> USD
        self.mock_data_manager.get_exchange_rate.side_effect = lambda f, t: Decimal("1.1") if f == Currency.EUR and t == Currency.USD else None

        value = self.analyzer._calculate_portfolio_value(portfolio)

        # 1000 + (500 * 1.1) = 1550
        assert value == Decimal("1550")
