"""Test transaction current balance functionality."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.portfolio.manager import PortfolioManager
from src.portfolio.models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    Portfolio,
    Transaction,
    TransactionType,
)


class TestTransactionCurrentBalance:
    """Test that transactions properly track current balance."""

    def test_deposit_sets_current_balance(self):
        """Test that deposit transaction sets correct current balance."""
        # Create a mock portfolio manager
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Use the manager to add the transaction
        result = manager.deposit_cash(
            amount=Decimal("500"), currency=Currency.USD, notes="Test deposit"
        )

        assert result is True

        # Get the transaction that was added
        transaction = portfolio.transactions[-1]

        # Verify the transaction has the correct current balance
        # For DEPOSIT: total_value = 500, so balance = 1000 + 500 = 1500
        expected_balance = Decimal("1500")
        assert transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_buy_sets_current_balance(self):
        """Test that buy transaction sets correct current balance."""
        # Create a mock portfolio manager
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Use the manager to add the transaction
        result = manager.buy_shares(
            symbol="AAPL",
            quantity=Decimal("10"),
            price=Decimal("50"),
            notes="Test buy",
        )

        assert result is True

        # Get the transaction that was added
        transaction = portfolio.transactions[-1]

        # Verify the transaction has the correct current balance
        # For BUY: total_value = (10 * 50) = 500, so balance = 1000 - 500 = 500
        expected_balance = Decimal("500")
        assert transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_chf_deposit_with_exchange_rate(self):
        """Test that CHF deposit properly tracks balance in CHF."""
        # Create a mock portfolio manager
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000"), Currency.CHF: Decimal("0")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Use the manager to add the transaction
        result = manager.deposit_cash(
            amount=Decimal("1000"), currency=Currency.CHF, notes="Test CHF deposit"
        )

        assert result is True

        # Get the transaction that was added
        transaction = portfolio.transactions[-1]

        # Verify the transaction has the correct current balance in CHF
        # For DEPOSIT: total_value = 1000, so balance = 0 + 1000 = 1000
        expected_balance = Decimal("1000")
        assert transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.CHF] == expected_balance
        # USD balance should remain unchanged
        assert portfolio.cash_balances[Currency.USD] == Decimal("1000")

    def test_sell_transaction_sets_current_balance(self):
        """Test that sell transaction sets correct current balance."""
        # Create a portfolio with existing position
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("100")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # First buy some shares to create a position
        manager.buy_shares(
            symbol="AAPL",
            quantity=Decimal("10"),
            price=Decimal("50"),
            notes="Initial buy",
        )

        # Now sell some shares
        result = manager.sell_shares(
            symbol="AAPL",
            quantity=Decimal("5"),
            price=Decimal("60"),
            notes="Test sell",
        )

        assert result is True

        # Get the sell transaction
        sell_transaction = portfolio.transactions[-1]
        assert sell_transaction.transaction_type == TransactionType.SELL

        # Verify the sell transaction has the correct current balance
        # Initial balance: 100
        # After buy: 100 - (10 * 50) = 100 - 500 = -400
        # After sell: -400 + (5 * 60) = -400 + 300 = -100
        expected_balance = Decimal("-100")
        assert sell_transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_dividend_transaction_sets_current_balance(self):
        """Test that dividend transaction sets correct current balance."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Add dividend transaction
        result = manager.add_dividend(
            symbol="AAPL", amount=Decimal("25.50"), notes="Quarterly dividend"
        )

        assert result is True

        # Get the dividend transaction
        dividend_transaction = portfolio.transactions[-1]
        assert dividend_transaction.transaction_type == TransactionType.DIVIDEND

        # Verify the dividend transaction has the correct current balance
        # For DIVIDEND: total_value = 25.50, so balance = 1000 + 25.50 = 1025.50
        expected_balance = Decimal("1025.50")
        assert dividend_transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_interest_transaction_sets_current_balance(self):
        """Test that interest transaction sets correct current balance."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("500")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Add interest transaction
        result = manager.add_transaction(
            symbol="CASH",
            transaction_type=TransactionType.INTEREST,
            quantity=Decimal("1"),
            price=Decimal("12.75"),
            currency=Currency.USD,
            notes="Monthly interest",
        )

        assert result is True

        # Get the interest transaction
        interest_transaction = portfolio.transactions[-1]
        assert interest_transaction.transaction_type == TransactionType.INTEREST

        # Verify the interest transaction has the correct current balance
        # For INTEREST: total_value = 12.75, so balance = 500 + 12.75 = 512.75
        expected_balance = Decimal("512.75")
        assert interest_transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_withdrawal_transaction_sets_current_balance(self):
        """Test that withdrawal transaction sets correct current balance."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Add withdrawal transaction
        result = manager.withdraw_cash(
            amount=Decimal("200"), currency=Currency.USD, notes="Emergency withdrawal"
        )

        assert result is True

        # Get the withdrawal transaction
        withdrawal_transaction = portfolio.transactions[-1]
        assert withdrawal_transaction.transaction_type == TransactionType.WITHDRAWAL

        # Verify the withdrawal transaction has the correct current balance
        # For WITHDRAWAL: total_value = 200, so balance = 1000 - 200 = 800
        expected_balance = Decimal("800")
        assert withdrawal_transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_multiple_currency_transactions(self):
        """Test that multiple currency transactions maintain correct balances."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={
                Currency.USD: Decimal("1000"),
                Currency.EUR: Decimal("500"),
                Currency.CHF: Decimal("0"),
            },
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Add multiple transactions in different currencies
        manager.deposit_cash(
            amount=Decimal("100"), currency=Currency.EUR, notes="EUR deposit"
        )
        manager.withdraw_cash(
            amount=Decimal("50"), currency=Currency.USD, notes="USD withdrawal"
        )
        manager.deposit_cash(
            amount=Decimal("200"), currency=Currency.CHF, notes="CHF deposit"
        )

        # Verify all transactions have correct current_balance
        transactions = portfolio.transactions
        assert len(transactions) == 3

        # EUR deposit: 500 + 100 = 600
        assert transactions[0].current_balance == Decimal("600")
        assert portfolio.cash_balances[Currency.EUR] == Decimal("600")

        # USD withdrawal: 1000 - 50 = 950
        assert transactions[1].current_balance == Decimal("950")
        assert portfolio.cash_balances[Currency.USD] == Decimal("950")

        # CHF deposit: 0 + 200 = 200
        assert transactions[2].current_balance == Decimal("200")
        assert portfolio.cash_balances[Currency.CHF] == Decimal("200")

    def test_transaction_without_fees(self):
        """Test that transactions without fees calculate balance correctly."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Buy shares without fees
        result = manager.buy_shares(
            symbol="TSLA",
            quantity=Decimal("5"),
            price=Decimal("100"),
            notes="Commission-free trade",
        )

        assert result is True

        # Get the transaction
        transaction = portfolio.transactions[-1]

        # Verify the transaction has the correct current balance
        # For BUY: total_value = (5 * 100) = 500, so balance = 1000 - 500 = 500
        expected_balance = Decimal("500")
        assert transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_transaction_sequence_balance_consistency(self):
        """Test that a sequence of transactions maintains consistent balance tracking."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Execute a sequence of transactions
        transactions = [
            ("deposit", Decimal("500"), Decimal("1500")),
            ("buy", Decimal("200"), Decimal("1300")),
            ("dividend", Decimal("25"), Decimal("1325")),
            ("sell", Decimal("150"), Decimal("1475")),
            ("withdrawal", Decimal("100"), Decimal("1375")),
        ]

        for i, (txn_type, amount, expected_balance) in enumerate(transactions):
            if txn_type == "deposit":
                result = manager.deposit_cash(amount=amount, notes=f"Transaction {i+1}")
            elif txn_type == "buy":
                result = manager.buy_shares(
                    symbol="STOCK",
                    quantity=Decimal("1"),
                    price=amount,
                    notes=f"Transaction {i+1}",
                )
            elif txn_type == "dividend":
                result = manager.add_dividend(
                    symbol="STOCK", amount=amount, notes=f"Transaction {i+1}"
                )
            elif txn_type == "sell":
                result = manager.sell_shares(
                    symbol="STOCK",
                    quantity=Decimal("1"),
                    price=amount,
                    notes=f"Transaction {i+1}",
                )
            elif txn_type == "withdrawal":
                result = manager.withdraw_cash(
                    amount=amount, notes=f"Transaction {i+1}"
                )

            assert result is True

            # Verify the transaction has the expected current balance
            transaction = portfolio.transactions[-1]
            assert (
                transaction.current_balance == expected_balance
            ), f"Transaction {i+1} failed: expected {expected_balance}, got {transaction.current_balance}"
            assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_edge_case_zero_amount_transaction(self):
        """Test edge case with zero amount transaction."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Add a zero amount deposit (edge case)
        result = manager.deposit_cash(
            amount=Decimal("0"), currency=Currency.USD, notes="Zero deposit"
        )

        assert result is True

        # Get the transaction
        transaction = portfolio.transactions[-1]

        # Verify the transaction has the correct current balance
        # For zero deposit: total_value = 0, so balance = 1000 + 0 = 1000
        expected_balance = Decimal("1000")
        assert transaction.current_balance == expected_balance
        assert portfolio.cash_balances[Currency.USD] == expected_balance

    def test_get_cash_balance_method(self):
        """Test the get_cash_balance helper method."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000"), Currency.CHF: Decimal("500")},
        )

        assert portfolio.get_cash_balance(Currency.USD) == Decimal("1000")
        assert portfolio.get_cash_balance(Currency.CHF) == Decimal("500")
        assert portfolio.get_cash_balance(Currency.EUR) == Decimal(
            "0"
        )  # Non-existent currency

    def test_transaction_with_notes(self):
        """Test that transactions with notes preserve the current_balance field."""
        portfolio = Portfolio(
            id="test",
            name="Test Portfolio",
            base_currency=Currency.USD,
            cash_balances={Currency.USD: Decimal("1000")},
        )

        # Mock the dependencies
        mock_storage = Mock()
        mock_storage.save_portfolio = Mock()
        mock_data_manager = Mock()
        mock_data_manager.get_instrument_info = Mock(return_value=None)

        manager = PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)
        manager.current_portfolio = portfolio

        # Add transaction with detailed notes
        result = manager.deposit_cash(
            amount=Decimal("250"),
            currency=Currency.USD,
            notes="Salary deposit from employer XYZ Corp, reference #12345",
        )

        assert result is True

        # Get the transaction
        transaction = portfolio.transactions[-1]

        # Verify both the notes and current_balance are preserved
        assert (
            transaction.notes
            == "Salary deposit from employer XYZ Corp, reference #12345"
        )
        assert transaction.current_balance == Decimal("1250")
        assert portfolio.cash_balances[Currency.USD] == Decimal("1250")
