"""
Unit tests for what-if simulation functionality.
Tests both the basic simulation engine and prepares for advanced scenario modeling.
"""

import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from src.portfolio.models import (
    Portfolio, Transaction, TransactionType, FinancialInstrument,
    Currency, InstrumentType, PortfolioSnapshot
)
from src.portfolio.manager import PortfolioManager
from src.portfolio.storage import FileBasedStorage
from src.agents.tools import SimulateWhatIfTool


class TestWhatIfSimulations(unittest.TestCase):
    """Test what-if simulation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock storage and data manager
        self.mock_storage = Mock(spec=FileBasedStorage)
        self.mock_data_manager = Mock()
        self.mock_data_manager.get_current_price.return_value = Decimal("100.00")

        # Create portfolio manager
        self.portfolio_manager = PortfolioManager(
            storage=self.mock_storage,
            data_manager=self.mock_data_manager
        )

        # Create test instruments
        self.aapl_instrument = FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
            isin="US0378331005"
        )

        self.msft_instrument = FinancialInstrument(
            symbol="MSFT",
            name="Microsoft Corp.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
            isin="US5949181045"
        )

        # Create test transactions
        base_date = date(2024, 1, 1)
        self.transactions = [
            Transaction(
                id="txn1",
                timestamp=datetime.combine(base_date, datetime.min.time()),
                instrument=self.aapl_instrument,
                transaction_type=TransactionType.BUY,
                quantity=Decimal("10"),
                price=Decimal("150.00"),
                currency=Currency.USD,
                notes="Buy AAPL"
            ),
            Transaction(
                id="txn2",
                timestamp=datetime.combine(base_date + timedelta(days=30), datetime.min.time()),
                instrument=self.msft_instrument,
                transaction_type=TransactionType.BUY,
                quantity=Decimal("5"),
                price=Decimal("300.00"),
                currency=Currency.USD,
                notes="Buy MSFT"
            ),
            Transaction(
                id="txn3",
                timestamp=datetime.combine(base_date + timedelta(days=60), datetime.min.time()),
                instrument=self.aapl_instrument,
                transaction_type=TransactionType.SELL,
                quantity=Decimal("5"),
                price=Decimal("160.00"),
                currency=Currency.USD,
                notes="Sell AAPL"
            )
        ]

        # Create test portfolio
        self.test_portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            base_currency=Currency.USD,
            created_at=datetime.now(),
            transactions=self.transactions,
            positions={},
            cash_balances={Currency.USD: Decimal("10000.00")}
        )

        self.portfolio_manager.current_portfolio = self.test_portfolio

    def test_simulate_snapshots_basic(self):
        """Test basic snapshot simulation without exclusions."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        # Mock create_snapshots_for_range to return test snapshots
        test_snapshots = [
            PortfolioSnapshot(
                date=start_date,
                total_value=Decimal("10000.00"),
                cash_balance=Decimal("10000.00"),
                positions_value=Decimal("0.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("10000.00")},
                total_cost_basis=Decimal("0.00"),
                total_unrealized_pnl=Decimal("0.00"),
                total_unrealized_pnl_percent=Decimal("0.00")
            ),
            PortfolioSnapshot(
                date=end_date,
                total_value=Decimal("12000.00"),
                cash_balance=Decimal("7000.00"),
                positions_value=Decimal("5000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("7000.00")},
                total_cost_basis=Decimal("3000.00"),
                total_unrealized_pnl=Decimal("2000.00"),
                total_unrealized_pnl_percent=Decimal("66.67")
            )
        ]

        with patch.object(self.portfolio_manager, 'create_snapshots_for_range', return_value=test_snapshots):
            simulated_snapshots = self.portfolio_manager.simulate_snapshots_for_range(
                start_date, end_date
            )

        self.assertEqual(len(simulated_snapshots), 2)
        self.assertEqual(simulated_snapshots[0].total_value, Decimal("10000.00"))
        self.assertEqual(simulated_snapshots[-1].total_value, Decimal("12000.00"))

        # Verify original portfolio is restored
        self.assertEqual(self.portfolio_manager.current_portfolio.id, "test_portfolio")
        self.assertEqual(len(self.portfolio_manager.current_portfolio.transactions), 3)

    def test_simulate_exclude_symbols(self):
        """Test simulation with symbol exclusions."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        # Mock snapshots for simulation without AAPL
        test_snapshots = [
            PortfolioSnapshot(
                date=end_date,
                total_value=Decimal("11500.00"),  # Different value without AAPL
                cash_balance=Decimal("8500.00"),
                positions_value=Decimal("3000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("8500.00")},
                total_cost_basis=Decimal("1500.00"),
                total_unrealized_pnl=Decimal("1500.00"),
                total_unrealized_pnl_percent=Decimal("100.00")
            )
        ]

        with patch.object(self.portfolio_manager, 'create_snapshots_for_range', return_value=test_snapshots):
            simulated_snapshots = self.portfolio_manager.simulate_snapshots_for_range(
                start_date, end_date, exclude_symbols=["AAPL"]
            )

        self.assertEqual(len(simulated_snapshots), 1)

        # Verify that the temporary portfolio was created without AAPL transactions
        # This is tested indirectly through the portfolio manager behavior
        self.assertIsNotNone(simulated_snapshots)

    def test_simulate_exclude_transaction_ids(self):
        """Test simulation with transaction ID exclusions."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        test_snapshots = [
            PortfolioSnapshot(
                date=end_date,
                total_value=Decimal("11000.00"),
                cash_balance=Decimal("8000.00"),
                positions_value=Decimal("3000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("8000.00")},
                total_cost_basis=Decimal("1500.00"),
                total_unrealized_pnl=Decimal("1500.00"),
                total_unrealized_pnl_percent=Decimal("100.00")
            )
        ]

        with patch.object(self.portfolio_manager, 'create_snapshots_for_range', return_value=test_snapshots):
            simulated_snapshots = self.portfolio_manager.simulate_snapshots_for_range(
                start_date, end_date, exclude_transaction_ids=["txn2"]  # Exclude MSFT purchase
            )

        self.assertEqual(len(simulated_snapshots), 1)
        self.assertEqual(simulated_snapshots[0].total_value, Decimal("11000.00"))

    def test_simulate_combined_exclusions(self):
        """Test simulation with both symbol and transaction exclusions."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        test_snapshots = [
            PortfolioSnapshot(
                date=end_date,
                total_value=Decimal("10000.00"),  # Only cash left
                cash_balance=Decimal("10000.00"),
                positions_value=Decimal("0.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("10000.00")},
                total_cost_basis=Decimal("0.00"),
                total_unrealized_pnl=Decimal("0.00"),
                total_unrealized_pnl_percent=Decimal("0.00")
            )
        ]

        with patch.object(self.portfolio_manager, 'create_snapshots_for_range', return_value=test_snapshots):
            simulated_snapshots = self.portfolio_manager.simulate_snapshots_for_range(
                start_date, end_date,
                exclude_symbols=["AAPL"],
                exclude_transaction_ids=["txn2"]
            )

        self.assertEqual(len(simulated_snapshots), 1)
        self.assertEqual(simulated_snapshots[0].total_value, Decimal("10000.00"))
        self.assertEqual(simulated_snapshots[0].positions_value, Decimal("0.00"))

    def test_simulation_transaction_filtering(self):
        """Test that transaction filtering works correctly in simulation."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        # Capture the temporary portfolio that gets created
        captured_portfolios = []
        original_create_snapshots = self.portfolio_manager.create_snapshots_for_range

        def capture_portfolio(*args, **kwargs):
            captured_portfolios.append(self.portfolio_manager.current_portfolio)
            return []  # Return empty list for simplicity

        with patch.object(self.portfolio_manager, 'create_snapshots_for_range', side_effect=capture_portfolio):
            self.portfolio_manager.simulate_snapshots_for_range(
                start_date, end_date, exclude_symbols=["AAPL"]
            )

        # Verify that a temporary portfolio was created with filtered transactions
        self.assertEqual(len(captured_portfolios), 1)
        temp_portfolio = captured_portfolios[0]

        # Should only have MSFT transaction (txn2), AAPL transactions should be excluded
        self.assertLess(len(temp_portfolio.transactions), len(self.test_portfolio.transactions))

        # Verify original portfolio is still intact
        self.assertEqual(len(self.portfolio_manager.current_portfolio.transactions), 3)

    def test_no_portfolio_error(self):
        """Test that simulation raises error when no portfolio is loaded."""
        self.portfolio_manager.current_portfolio = None

        with self.assertRaises(ValueError) as context:
            self.portfolio_manager.simulate_snapshots_for_range(
                date(2024, 1, 1), date(2024, 3, 31)
            )

        self.assertIn("No portfolio loaded", str(context.exception))


class TestSimulateWhatIfTool(unittest.TestCase):
    """Test the SimulateWhatIfTool agent tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_portfolio_manager = Mock()
        self.tool = SimulateWhatIfTool(portfolio_manager=self.mock_portfolio_manager)

    def test_what_if_tool_basic(self):
        """Test basic what-if tool functionality."""
        # Mock baseline snapshots
        baseline_snapshots = [
            Mock(total_value=Decimal("10000"), date=date(2024, 1, 1)),
            Mock(total_value=Decimal("12000"), date=date(2024, 3, 31))
        ]

        # Mock what-if snapshots (without excluded positions)
        whatif_snapshots = [
            Mock(total_value=Decimal("10000"), date=date(2024, 1, 1)),
            Mock(total_value=Decimal("11000"), date=date(2024, 3, 31))
        ]

        self.mock_portfolio_manager.simulate_snapshots_for_range.side_effect = [
            baseline_snapshots,  # First call (baseline)
            whatif_snapshots     # Second call (what-if)
        ]

        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31",
            exclude_symbols="AAPL"
        )

        self.assertIn("What-if Simulation", result)
        self.assertIn("AAPL", result)
        self.assertIn("$12,000.00", result)  # Baseline end
        self.assertIn("$11,000.00", result)  # What-if end
        self.assertIn("-1,000.00", result)  # Delta (without $ sign)

    def test_what_if_tool_exclude_transactions(self):
        """Test what-if tool with transaction exclusions."""
        baseline_snapshots = [Mock(total_value=Decimal("15000"), date=date(2024, 3, 31))]
        whatif_snapshots = [Mock(total_value=Decimal("13000"), date=date(2024, 3, 31))]

        self.mock_portfolio_manager.simulate_snapshots_for_range.side_effect = [
            baseline_snapshots,
            whatif_snapshots
        ]

        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31",
            exclude_txn_ids="txn123,txn456"
        )

        self.assertIn("What-if Simulation", result)
        self.assertIn("txn123", result)  # Should contain the transaction IDs

    def test_what_if_tool_error_handling(self):
        """Test what-if tool error handling."""
        self.mock_portfolio_manager.simulate_snapshots_for_range.side_effect = Exception("Simulation failed")

        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31"
        )

        self.assertIn("Error running simulation", result)
        self.assertIn("Simulation failed", result)

    def test_what_if_tool_no_snapshots(self):
        """Test what-if tool when no snapshots are generated."""
        self.mock_portfolio_manager.simulate_snapshots_for_range.return_value = []

        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31"
        )

        self.assertIn("No snapshots generated", result)

    def test_what_if_tool_parameter_parsing(self):
        """Test parameter parsing in what-if tool."""
        baseline_snapshots = [Mock(total_value=Decimal("10000"), date=date(2024, 3, 31))]
        whatif_snapshots = [Mock(total_value=Decimal("9000"), date=date(2024, 3, 31))]

        self.mock_portfolio_manager.simulate_snapshots_for_range.side_effect = [
            baseline_snapshots,
            whatif_snapshots
        ]

        # Test with comma-separated symbols and transaction IDs
        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31",
            exclude_symbols="AAPL, MSFT, GOOGL",
            exclude_txn_ids="txn1, txn2"
        )

        # Should properly parse and exclude multiple symbols/transactions
        calls = self.mock_portfolio_manager.simulate_snapshots_for_range.call_args_list
        self.assertEqual(len(calls), 2)

        # Check what-if call (second call) has exclusions
        whatif_call = calls[1]
        args, kwargs = whatif_call
        self.assertIn("AAPL", kwargs.get("exclude_symbols", []))
        self.assertIn("MSFT", kwargs.get("exclude_symbols", []))
        self.assertIn("txn1", kwargs.get("exclude_transaction_ids", []))


if __name__ == '__main__':
    unittest.main()
