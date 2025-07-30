"""
Unit tests for portfolio models.
"""

import pytest
from datetime import datetime, date
from decimal import Decimal

try:
    from src.portfolio.models import (
        Currency,
        FinancialInstrument,
        InstrumentType,
        Portfolio,
        Position,
        Transaction,
        TransactionType,
        PortfolioSnapshot,
    )
except ImportError:
    from portfolio.models import (
        Currency,
        FinancialInstrument,
        InstrumentType,
        Portfolio,
        Position,
        Transaction,
        TransactionType,
        PortfolioSnapshot,
    )


class TestFinancialInstrument:
    """Test cases for FinancialInstrument model."""

    def test_create_stock_instrument(self):
        """Test creating a stock instrument."""
        instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
            exchange="NASDAQ",
            isin="US0378331005",
        )

        assert instrument.symbol == "AAPL"
        assert instrument.name == "Apple Inc."
        assert instrument.instrument_type == InstrumentType.STOCK
        assert instrument.currency == Currency.USD
        assert instrument.exchange == "NASDAQ"
        assert instrument.isin == "US0378331005"

    def test_create_crypto_instrument(self):
        """Test creating a cryptocurrency instrument."""
        instrument = FinancialInstrument(
            symbol="BTC-USD",
            name="Bitcoin",
            instrument_type=InstrumentType.CRYPTO,
            currency=Currency.USD,
        )

        assert instrument.symbol == "BTC-USD"
        assert instrument.name == "Bitcoin"
        assert instrument.instrument_type == InstrumentType.CRYPTO
        assert instrument.currency == Currency.USD
        assert instrument.exchange is None
        assert instrument.isin is None

    def test_instrument_validation(self):
        """Test instrument validation."""
        with pytest.raises(ValueError):
            FinancialInstrument(
                symbol="",  # Empty symbol should fail
                name="Test",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
            )


class TestTransaction:
    """Test cases for Transaction model."""

    def test_create_buy_transaction(self, sample_instrument):
        """Test creating a buy transaction."""
        transaction = Transaction(
            id="txn-123",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            fees=Decimal("1.00"),
            currency=Currency.USD,
            notes="Test buy transaction",
        )

        assert transaction.id == "txn-123"
        assert transaction.transaction_type == TransactionType.BUY
        assert transaction.quantity == Decimal("100")
        assert transaction.price == Decimal("150.00")
        assert transaction.fees == Decimal("1.00")
        assert transaction.total_value == Decimal("15001.00")  # 100 * 150 + 1

    def test_create_sell_transaction(self, sample_instrument):
        """Test creating a sell transaction."""
        transaction = Transaction(
            id="txn-456",
            timestamp=datetime(2024, 1, 20, 14, 15, 0),
            instrument=sample_instrument,
            transaction_type=TransactionType.SELL,
            quantity=Decimal("50"),
            price=Decimal("160.00"),
            fees=Decimal("1.00"),
            currency=Currency.USD,
        )

        assert transaction.transaction_type == TransactionType.SELL
        assert transaction.total_value == Decimal("7999.00")  # 50 * 160 - 1

    def test_dividend_transaction(self, sample_instrument):
        """Test creating a dividend transaction."""
        transaction = Transaction(
            id="txn-div",
            timestamp=datetime(2024, 1, 25, 9, 0, 0),
            instrument=sample_instrument,
            transaction_type=TransactionType.DIVIDEND,
            quantity=Decimal("1"),
            price=Decimal("25.00"),
            fees=Decimal("0"),
            currency=Currency.USD,
        )

        assert transaction.transaction_type == TransactionType.DIVIDEND
        assert transaction.total_value == Decimal("25.00")

    def test_transaction_total_value_calculation(self, sample_instrument):
        """Test transaction total value calculation for different types."""
        # Buy transaction: base value + fees
        buy_txn = Transaction(
            id="buy",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("10"),
            price=Decimal("100"),
            fees=Decimal("5"),
            currency=Currency.USD,
        )
        assert buy_txn.total_value == Decimal("1005")

        # Sell transaction: base value - fees
        sell_txn = Transaction(
            id="sell",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.SELL,
            quantity=Decimal("10"),
            price=Decimal("100"),
            fees=Decimal("5"),
            currency=Currency.USD,
        )
        assert sell_txn.total_value == Decimal("995")


class TestPosition:
    """Test cases for Position model."""

    def test_create_position(self, sample_instrument):
        """Test creating a position."""
        position = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00"),
        )

        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("150.00")
        assert position.current_price == Decimal("160.00")

    def test_position_calculations(self, sample_instrument):
        """Test position value calculations."""
        position = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
            current_price=Decimal("160.00"),
        )

        assert position.cost_basis == Decimal("15000.00")  # 100 * 150
        assert position.market_value == Decimal("16000.00")  # 100 * 160
        assert position.unrealized_pnl == Decimal("1000.00")  # 16000 - 15000
        assert position.unrealized_pnl_percent == Decimal(
            "6.666666666666666666666666667"
        )

    def test_position_without_current_price(self, sample_instrument):
        """Test position calculations when current price is not available."""
        position = Position(
            instrument=sample_instrument,
            quantity=Decimal("100"),
            average_cost=Decimal("150.00"),
        )

        assert position.cost_basis == Decimal("15000.00")
        assert position.market_value is None
        assert position.unrealized_pnl is None
        assert position.unrealized_pnl_percent is None


class TestPortfolio:
    """Test cases for Portfolio model."""

    def test_create_empty_portfolio(self):
        """Test creating an empty portfolio."""
        portfolio = Portfolio(
            id="test-123",
            name="Test Portfolio",
            base_currency=Currency.USD,
        )

        assert portfolio.id == "test-123"
        assert portfolio.name == "Test Portfolio"
        assert portfolio.base_currency == Currency.USD
        assert len(portfolio.transactions) == 0
        assert len(portfolio.positions) == 0
        assert len(portfolio.cash_balances) == 0

    def test_add_buy_transaction(self, sample_portfolio, sample_buy_transaction):
        """Test adding a buy transaction to portfolio."""
        initial_position_count = len(sample_portfolio.positions)
        sample_portfolio.add_transaction(sample_buy_transaction)

        assert len(sample_portfolio.transactions) == 1
        assert len(sample_portfolio.positions) == initial_position_count + 1

        position = sample_portfolio.positions["AAPL"]
        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("150.00")

    def test_add_multiple_buy_transactions(self, sample_portfolio, sample_instrument):
        """Test adding multiple buy transactions for the same instrument."""
        # First buy
        txn1 = Transaction(
            id="txn1",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            fees=Decimal("1.00"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(txn1)

        # Second buy at different price
        txn2 = Transaction(
            id="txn2",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("50"),
            price=Decimal("160.00"),
            fees=Decimal("1.00"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(txn2)

        position = sample_portfolio.positions["AAPL"]
        assert position.quantity == Decimal("150")  # 100 + 50
        # Average cost: (100*150 + 1 + 50*160 + 1) / 150 = 23002 / 150 = 153.346...
        expected_avg = Decimal("23002") / Decimal("150")
        assert position.average_cost == expected_avg

    def test_add_sell_transaction(self, sample_portfolio, sample_instrument):
        """Test adding sell transaction reduces position."""
        # First buy
        buy_txn = Transaction(
            id="buy",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            fees=Decimal("0"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(buy_txn)

        # Then sell part
        sell_txn = Transaction(
            id="sell",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.SELL,
            quantity=Decimal("30"),
            price=Decimal("160.00"),
            fees=Decimal("0"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(sell_txn)

        position = sample_portfolio.positions["AAPL"]
        assert position.quantity == Decimal("70")  # 100 - 30
        assert position.average_cost == Decimal("150.00")  # Should remain the same

    def test_sell_entire_position(self, sample_portfolio, sample_instrument):
        """Test selling entire position removes it from portfolio."""
        # Buy
        buy_txn = Transaction(
            id="buy",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            fees=Decimal("0"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(buy_txn)

        # Sell all
        sell_txn = Transaction(
            id="sell",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.SELL,
            quantity=Decimal("100"),
            price=Decimal("160.00"),
            fees=Decimal("0"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(sell_txn)

        assert "AAPL" not in sample_portfolio.positions

    def test_cash_transactions(self, sample_portfolio):
        """Test cash deposit and withdrawal transactions."""
        # Initial cash should be empty
        assert len(sample_portfolio.cash_balances) == 0

        # Add deposit transaction
        deposit_instrument = FinancialInstrument(
            symbol="CASH",
            name="Cash",
            instrument_type=InstrumentType.CASH,
            currency=Currency.USD,
        )

        deposit_txn = Transaction(
            id="deposit",
            timestamp=datetime.now(),
            instrument=deposit_instrument,
            transaction_type=TransactionType.DEPOSIT,
            quantity=Decimal("1"),
            price=Decimal("1000.00"),
            fees=Decimal("0"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(deposit_txn)

        assert sample_portfolio.cash_balances[Currency.USD] == Decimal("1000.00")

        # Add withdrawal
        withdrawal_txn = Transaction(
            id="withdrawal",
            timestamp=datetime.now(),
            instrument=deposit_instrument,
            transaction_type=TransactionType.WITHDRAWAL,
            quantity=Decimal("1"),
            price=Decimal("200.00"),
            fees=Decimal("0"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(withdrawal_txn)

        assert sample_portfolio.cash_balances[Currency.USD] == Decimal("800.00")

    def test_get_positions_by_type(self, sample_portfolio):
        """Test filtering positions by instrument type."""
        # Add stock
        stock_instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        # Add crypto
        crypto_instrument = FinancialInstrument(
            symbol="BTC",
            name="Bitcoin",
            instrument_type=InstrumentType.CRYPTO,
            currency=Currency.USD,
        )

        # Add transactions
        for instrument in [stock_instrument, crypto_instrument]:
            txn = Transaction(
                id=f"txn-{instrument.symbol}",
                timestamp=datetime.now(),
                instrument=instrument,
                transaction_type=TransactionType.BUY,
                quantity=Decimal("10"),
                price=Decimal("100"),
                fees=Decimal("0"),
                currency=Currency.USD,
            )
            sample_portfolio.add_transaction(txn)

        stock_positions = sample_portfolio.get_positions_by_type(InstrumentType.STOCK)
        crypto_positions = sample_portfolio.get_positions_by_type(InstrumentType.CRYPTO)

        assert len(stock_positions) == 1
        assert len(crypto_positions) == 1
        assert stock_positions[0].instrument.symbol == "AAPL"
        assert crypto_positions[0].instrument.symbol == "BTC"

    def test_get_total_value_single_currency(self, sample_portfolio, sample_instrument):
        """Test calculating total portfolio value in single currency."""
        # Add some cash
        sample_portfolio.cash_balances[Currency.USD] = Decimal("5000.00")

        # Add a position
        txn = Transaction(
            id="txn",
            timestamp=datetime.now(),
            instrument=sample_instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            fees=Decimal("0"),
            currency=Currency.USD,
        )
        sample_portfolio.add_transaction(txn)

        # Update current price
        sample_portfolio.positions["AAPL"].current_price = Decimal("160.00")

        total_value = sample_portfolio.get_total_value()
        expected = Decimal("5000.00") + Decimal("16000.00")  # cash + position value
        assert total_value == expected

    def test_get_total_value_with_exchange_rates(self, sample_portfolio):
        """Test calculating total portfolio value with currency conversion."""
        # Add cash in different currencies
        sample_portfolio.cash_balances[Currency.USD] = Decimal("1000.00")
        sample_portfolio.cash_balances[Currency.EUR] = Decimal("500.00")

        # Provide exchange rates
        exchange_rates = {"EUR": Decimal("1.10")}  # 1 EUR = 1.10 USD

        total_value = sample_portfolio.get_total_value(exchange_rates)
        expected = Decimal("1000.00") + (Decimal("500.00") * Decimal("1.10"))
        assert total_value == expected


class TestPortfolioSnapshot:
    """Test cases for PortfolioSnapshot model."""

    def test_create_snapshot(self):
        """Test creating a portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            date=date(2024, 1, 15),
            total_value=Decimal("25000.00"),
            cash_balance=Decimal("5000.00"),
            positions_value=Decimal("20000.00"),
            base_currency=Currency.USD,
        )

        assert snapshot.date == date(2024, 1, 15)
        assert snapshot.total_value == Decimal("25000.00")
        assert snapshot.cash_balance == Decimal("5000.00")
        assert snapshot.positions_value == Decimal("20000.00")
        assert snapshot.base_currency == Currency.USD

    def test_snapshot_validation(self):
        """Test snapshot validation."""
        # Total value should equal cash + positions
        snapshot = PortfolioSnapshot(
            date=date(2024, 1, 15),
            total_value=Decimal("25000.00"),
            cash_balance=Decimal("5000.00"),
            positions_value=Decimal("20000.00"),
            base_currency=Currency.USD,
        )

        assert snapshot.cash_balance + snapshot.positions_value == snapshot.total_value


class TestEnums:
    """Test cases for enumeration classes."""

    def test_currency_enum(self):
        """Test Currency enum values."""
        assert Currency.USD.value == "USD"
        assert Currency.EUR.value == "EUR"
        assert Currency.BTC.value == "BTC"

        # Test that we can create from string
        assert Currency("USD") == Currency.USD

    def test_instrument_type_enum(self):
        """Test InstrumentType enum values."""
        assert InstrumentType.STOCK.value == "stock"
        assert InstrumentType.CRYPTO.value == "crypto"
        assert InstrumentType.BOND.value == "bond"

    def test_transaction_type_enum(self):
        """Test TransactionType enum values."""
        assert TransactionType.BUY.value == "buy"
        assert TransactionType.SELL.value == "sell"
        assert TransactionType.DIVIDEND.value == "dividend"
