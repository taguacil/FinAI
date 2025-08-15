"""Test portfolio models functionality."""

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.portfolio.models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    Portfolio,
    PortfolioSnapshot,
    Position,
    Transaction,
    TransactionType,
)


class TestPosition:
    """Test Position model functionality."""

    def test_position_creation(self):
        """Test basic position creation."""
        instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        position = Position(
            instrument=instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.50"),
        )

        assert position.instrument.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("150.50")
        assert position.current_price is None
        assert position.last_updated is None

    def test_position_market_value_calculation(self):
        """Test market value calculation."""
        instrument = FinancialInstrument(
            symbol="TSLA",
            name="Tesla Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        position = Position(
            instrument=instrument,
            quantity=Decimal("50"),
            average_cost=Decimal("200.00"),
            current_price=Decimal("250.00"),
        )

        expected_market_value = Decimal("50") * Decimal("250.00")
        assert position.market_value == expected_market_value

    def test_position_cost_basis_calculation(self):
        """Test cost basis calculation."""
        instrument = FinancialInstrument(
            symbol="MSFT",
            name="Microsoft Corporation",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        position = Position(
            instrument=instrument,
            quantity=Decimal("75"),
            average_cost=Decimal("300.00"),
        )

        expected_cost_basis = Decimal("75") * Decimal("300.00")
        assert position.cost_basis == expected_cost_basis

    def test_position_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation."""
        instrument = FinancialInstrument(
            symbol="GOOGL",
            name="Alphabet Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        position = Position(
            instrument=instrument,
            quantity=Decimal("25"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("120.00"),
        )

        expected_pnl = (Decimal("25") * Decimal("120.00")) - (
            Decimal("25") * Decimal("100.00")
        )
        assert position.unrealized_pnl == expected_pnl
        assert position.unrealized_pnl == Decimal("500.00")

    def test_position_unrealized_pnl_percentage(self):
        """Test unrealized P&L percentage calculation."""
        instrument = FinancialInstrument(
            symbol="AMZN",
            name="Amazon.com Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        position = Position(
            instrument=instrument,
            quantity=Decimal("10"),
            average_cost=Decimal("100.00"),
            current_price=Decimal("110.00"),
        )

        expected_percentage = Decimal("10.00")  # 10% gain
        assert position.unrealized_pnl_percent == expected_percentage

    def test_position_with_no_current_price(self):
        """Test position behavior when current price is not set."""
        instrument = FinancialInstrument(
            symbol="NFLX",
            name="Netflix Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        position = Position(
            instrument=instrument,
            quantity=Decimal("30"),
            average_cost=Decimal("400.00"),
        )

        assert position.market_value is None
        assert position.unrealized_pnl is None
        assert position.unrealized_pnl_percent is None

    def test_position_zero_quantity(self):
        """Test position with zero quantity."""
        instrument = FinancialInstrument(
            symbol="META",
            name="Meta Platforms Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        position = Position(
            instrument=instrument, quantity=Decimal("0"), average_cost=Decimal("200.00")
        )

        assert position.cost_basis == Decimal("0")
        # When quantity is 0, market_value should be 0 regardless of current_price
        assert position.market_value == Decimal("0")


class TestPortfolio:
    """Test Portfolio model functionality."""

    def test_portfolio_creation(self):
        """Test basic portfolio creation."""
        portfolio = Portfolio(
            id="test_portfolio", name="Test Portfolio", base_currency=Currency.USD
        )

        assert portfolio.id == "test_portfolio"
        assert portfolio.name == "Test Portfolio"
        assert portfolio.base_currency == Currency.USD
        assert len(portfolio.transactions) == 0
        assert len(portfolio.positions) == 0
        assert len(portfolio.cash_balances) == 0

    def test_portfolio_with_initial_cash(self):
        """Test portfolio creation with initial cash balances."""
        portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={
                Currency.USD: Decimal("10000"),
                Currency.EUR: Decimal("5000"),
            },
        )

        assert portfolio.cash_balances[Currency.USD] == Decimal("10000")
        assert portfolio.cash_balances[Currency.EUR] == Decimal("5000")

    def test_portfolio_get_cash_balance(self):
        """Test get_cash_balance method."""
        portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000"), Currency.CHF: Decimal("500")},
        )

        assert portfolio.get_cash_balance(Currency.USD) == Decimal("1000")
        assert portfolio.get_cash_balance(Currency.CHF) == Decimal("500")
        assert portfolio.get_cash_balance(Currency.EUR) == Decimal("0")  # Non-existent

    def test_portfolio_get_positions_by_type(self):
        """Test get_positions_by_type method."""
        portfolio = Portfolio(
            id="test_portfolio", name="Test Portfolio", base_currency=Currency.USD
        )

        # Create some positions
        stock_instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        bond_instrument = FinancialInstrument(
            symbol="BOND",
            name="Government Bond",
            instrument_type=InstrumentType.BOND,
            currency=Currency.USD,
        )

        stock_position = Position(
            instrument=stock_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
        )

        bond_position = Position(
            instrument=bond_instrument,
            quantity=Decimal("10"),
            average_cost=Decimal("1000.00"),
        )

        portfolio.positions = {"AAPL": stock_position, "BOND": bond_position}

        stock_positions = portfolio.get_positions_by_type(InstrumentType.STOCK)
        bond_positions = portfolio.get_positions_by_type(InstrumentType.BOND)

        assert len(stock_positions) == 1
        assert stock_positions[0].instrument.symbol == "AAPL"

        assert len(bond_positions) == 1
        assert bond_positions[0].instrument.symbol == "BOND"

    def test_portfolio_total_value_calculation(self):
        """Test total value calculation with exchange rates."""
        portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000"), Currency.EUR: Decimal("500")},
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
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00"),
        )

        portfolio.positions["AAPL"] = position

        # Test with exchange rates
        exchange_rates = {"EUR": Decimal("1.20")}  # 1 EUR = 1.20 USD

        total_value = portfolio.get_total_value(exchange_rates)

        # Expected: 1000 USD + (500 EUR * 1.20) + (10 * 160) = 1000 + 600 + 1600 = 3200
        expected_total = (
            Decimal("1000")
            + (Decimal("500") * Decimal("1.20"))
            + (Decimal("10") * Decimal("160.00"))
        )
        assert total_value == expected_total

    def test_portfolio_total_value_with_rate_function(self):
        """Test total value calculation with rate function."""
        portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000"), Currency.CHF: Decimal("500")},
        )

        # Mock rate function
        def mock_rate_function(from_currency, to_currency):
            if from_currency == Currency.CHF and to_currency == Currency.USD:
                return Decimal("1.10")  # 1 CHF = 1.10 USD
            return None

        total_value = portfolio.get_total_value_with_rate_function(mock_rate_function)

        # Expected: 1000 USD + (500 CHF * 1.10) = 1000 + 550 = 1550
        expected_total = Decimal("1000") + (Decimal("500") * Decimal("1.10"))
        assert total_value == expected_total


class TestPortfolioSnapshot:
    """Test PortfolioSnapshot model functionality."""

    def test_portfolio_snapshot_creation(self):
        """Test basic portfolio snapshot creation."""
        snapshot = PortfolioSnapshot(
            date=date(2024, 1, 1),
            total_value=Decimal("10000"),
            cash_balance=Decimal("2000"),
            positions_value=Decimal("8000"),
            base_currency=Currency.USD,
            total_cost_basis=Decimal("9000"),
            total_unrealized_pnl=Decimal("1000"),
            total_unrealized_pnl_percent=Decimal("11.11"),
        )

        assert snapshot.date == date(2024, 1, 1)
        assert snapshot.total_value == Decimal("10000")
        assert snapshot.cash_balance == Decimal("2000")
        assert snapshot.positions_value == Decimal("8000")
        assert snapshot.base_currency == Currency.USD
        assert snapshot.total_cost_basis == Decimal("9000")
        assert snapshot.total_unrealized_pnl == Decimal("1000")
        assert snapshot.total_unrealized_pnl_percent == Decimal("11.11")

    def test_portfolio_snapshot_with_positions(self):
        """Test portfolio snapshot with positions."""
        # Create positions
        instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        position = Position(
            instrument=instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00"),
        )

        snapshot = PortfolioSnapshot(
            date=date(2024, 1, 1),
            total_value=Decimal("16000"),
            cash_balance=Decimal("0"),
            positions_value=Decimal("16000"),
            base_currency=Currency.USD,
            total_cost_basis=Decimal("15000"),
            total_unrealized_pnl=Decimal("1000"),
            total_unrealized_pnl_percent=Decimal("6.67"),
            positions={"AAPL": position},
            cash_balances={Currency.USD: Decimal("0")},
        )

        assert len(snapshot.positions) == 1
        assert "AAPL" in snapshot.positions
        assert snapshot.positions["AAPL"].quantity == Decimal("100")
        assert snapshot.cash_balances[Currency.USD] == Decimal("0")


class TestFinancialInstrument:
    """Test FinancialInstrument model functionality."""

    def test_financial_instrument_creation(self):
        """Test basic financial instrument creation."""
        instrument = FinancialInstrument(
            symbol="BTC",
            name="Bitcoin",
            instrument_type=InstrumentType.CRYPTO,
            currency=Currency.BTC,
        )

        assert instrument.symbol == "BTC"
        assert instrument.name == "Bitcoin"
        assert instrument.instrument_type == InstrumentType.CRYPTO
        assert instrument.currency == Currency.BTC
        assert instrument.isin is None
        assert instrument.exchange is None

    def test_financial_instrument_with_optional_fields(self):
        """Test financial instrument with all optional fields."""
        instrument = FinancialInstrument(
            symbol="SPY",
            isin="US78462F1030",
            name="SPDR S&P 500 ETF Trust",
            instrument_type=InstrumentType.ETF,
            currency=Currency.USD,
            exchange="NYSE Arca",
        )

        assert instrument.symbol == "SPY"
        assert instrument.isin == "US78462F1030"
        assert instrument.name == "SPDR S&P 500 ETF Trust"
        assert instrument.instrument_type == InstrumentType.ETF
        assert instrument.currency == Currency.USD
        assert instrument.exchange == "NYSE Arca"

    def test_financial_instrument_validation(self):
        """Test financial instrument validation."""
        # Test that symbol cannot be empty
        with pytest.raises(ValueError):
            FinancialInstrument(
                symbol="",
                name="Test",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
            )

        # Test that name cannot be empty
        with pytest.raises(ValueError):
            FinancialInstrument(
                symbol="TEST",
                name="",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
            )


class TestTransaction:
    """Test Transaction model functionality."""

    def test_transaction_creation(self):
        """Test basic transaction creation."""
        instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        transaction = Transaction(
            id="txn_001",
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            instrument=instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            currency=Currency.USD,
            notes="Test purchase",
        )

        assert transaction.id == "txn_001"
        assert transaction.transaction_type == TransactionType.BUY
        assert transaction.quantity == Decimal("100")
        assert transaction.price == Decimal("150.00")
        assert transaction.currency == Currency.USD
        assert transaction.notes == "Test purchase"
        assert transaction.current_balance is None  # Not set yet

    def test_transaction_total_value_calculation(self):
        """Test transaction total value calculation."""
        instrument = FinancialInstrument(
            symbol="TSLA",
            name="Tesla Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        # Test BUY transaction
        buy_transaction = Transaction(
            id="buy_001",
            timestamp=datetime.now(),
            instrument=instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("50"),
            price=Decimal("200.00"),
            currency=Currency.USD,
        )

        # For BUY: total_value = (50 * 200) = 10000
        expected_buy_value = Decimal("50") * Decimal("200.00")
        assert buy_transaction.total_value == expected_buy_value

        # Test SELL transaction
        sell_transaction = Transaction(
            id="sell_001",
            timestamp=datetime.now(),
            instrument=instrument,
            transaction_type=TransactionType.SELL,
            quantity=Decimal("25"),
            price=Decimal("220.00"),
            currency=Currency.USD,
        )

        # For SELL: total_value = (25 * 220) = 5500
        expected_sell_value = Decimal("25") * Decimal("220.00")
        assert sell_transaction.total_value == expected_sell_value

        # Test DEPOSIT transaction
        deposit_transaction = Transaction(
            id="deposit_001",
            timestamp=datetime.now(),
            instrument=instrument,
            transaction_type=TransactionType.DEPOSIT,
            quantity=Decimal("1"),
            price=Decimal("1000.00"),
            currency=Currency.USD,
        )

        # For DEPOSIT: total_value = (1 * 1000) = 1000
        expected_deposit_value = Decimal("1") * Decimal("1000.00")
        assert deposit_transaction.total_value == expected_deposit_value

    def test_transaction_with_current_balance(self):
        """Test transaction with current_balance field."""
        instrument = FinancialInstrument(
            symbol="MSFT",
            name="Microsoft Corporation",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        transaction = Transaction(
            id="txn_002",
            timestamp=datetime.now(),
            instrument=instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("10"),
            price=Decimal("300.00"),
            currency=Currency.USD,
            current_balance=Decimal("7000.00"),  # Set manually for testing
        )

        assert transaction.current_balance == Decimal("7000.00")
        assert transaction.total_value == Decimal("3000.00")  # (10 * 300)
