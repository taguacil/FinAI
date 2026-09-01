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

    def test_calculate_portfolio_value_with_position(self):
        """Test total portfolio value including cash and a position."""
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

        # Same currency -> exchange rate not needed, but stub anyway
        self.mock_data_manager.get_exchange_rate.return_value = Decimal("1.0")

        total_value = self.analyzer._calculate_portfolio_value(portfolio)
        cash_balance = self.analyzer._calculate_cash_balance(portfolio)

        # 1000 cash + (10 * 160) position = 2600
        assert total_value == Decimal("1000") + (Decimal("10") * Decimal("160"))
        assert total_value == Decimal("2600")
        assert cash_balance == Decimal("1000")

    def test_get_external_cash_flows_by_day(self):
        """Test external cash flow aggregation from deposits/withdrawals."""
        portfolio = Portfolio(
            id="test-id",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("10000")},
        )

        cash_instrument = FinancialInstrument(
            symbol="USD",
            name="US Dollar",
            instrument_type=InstrumentType.CASH,
            currency=Currency.USD,
        )

        deposit_date = datetime.now() - timedelta(days=5)
        withdrawal_date = datetime.now() - timedelta(days=2)

        deposit = Transaction(
            id="dep1",
            instrument=cash_instrument,
            transaction_type=TransactionType.DEPOSIT,
            quantity=Decimal("1000"),
            price=Decimal("1"),
            timestamp=deposit_date,
            currency=Currency.USD,
        )
        withdrawal = Transaction(
            id="wd1",
            instrument=cash_instrument,
            transaction_type=TransactionType.WITHDRAWAL,
            quantity=Decimal("300"),
            price=Decimal("1"),
            timestamp=withdrawal_date,
            currency=Currency.USD,
        )
        portfolio.add_transaction(deposit)
        portfolio.add_transaction(withdrawal)

        start_date = date.today() - timedelta(days=6)
        end_date = date.today()

        flows = self.analyzer.get_external_cash_flows_by_day(portfolio, start_date, end_date)

        # Same currency: no FX conversion required
        self.mock_data_manager.get_historical_fx_rate_on.assert_not_called()

        assert flows[deposit_date.date()] == Decimal("1000")
        assert flows[withdrawal_date.date()] == Decimal("-300")

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
