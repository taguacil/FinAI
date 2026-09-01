"""
Unit tests for what-if simulation functionality.
Tests both the basic simulation engine and prepares for advanced scenario modeling.
"""

import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd

from src.portfolio.models import (
    Portfolio, Transaction, TransactionType, FinancialInstrument,
    Currency, InstrumentType
)
from src.portfolio.manager import PortfolioManager
from src.portfolio.storage import FileBasedStorage
from src.agents.tools import SimulateWhatIfTool


def _make_history_df(values):
    """Build a value-history DataFrame like PortfolioHistory.get_value_history returns."""
    return pd.DataFrame(
        {
            "total_value": [float(v) for v in values],
            "cash_value": [0.0 for _ in values],
            "positions_value": [float(v) for v in values],
        }
    )


class TestWhatIfSimulations(unittest.TestCase):
    """Test what-if simulation functionality via PortfolioManager.simulate_portfolio_history."""

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
        """Test basic history simulation without exclusions returns the value history unchanged."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        history_df = _make_history_df([10000.00, 12000.00])

        # PortfolioHistory is imported lazily inside simulate_portfolio_history.
        with patch("src.portfolio.portfolio_history.PortfolioHistory") as mock_ph:
            mock_ph.return_value.get_value_history.return_value = history_df
            result = self.portfolio_manager.simulate_portfolio_history(start_date, end_date)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["total_value"].iloc[0], 10000.00)
        self.assertEqual(result["total_value"].iloc[-1], 12000.00)

        # With no exclusions, the temporary portfolio keeps all transactions.
        _, kwargs = mock_ph.call_args
        temp_portfolio = kwargs["portfolio"]
        self.assertEqual(len(temp_portfolio.transactions), 3)

        # Verify original portfolio is untouched (simulation must not mutate state).
        self.assertEqual(self.portfolio_manager.current_portfolio.id, "test_portfolio")
        self.assertEqual(len(self.portfolio_manager.current_portfolio.transactions), 3)

    def test_simulate_exclude_symbols(self):
        """Test simulation with symbol exclusions filters those transactions out."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        with patch("src.portfolio.portfolio_history.PortfolioHistory") as mock_ph:
            mock_ph.return_value.get_value_history.return_value = _make_history_df([11500.00])
            result = self.portfolio_manager.simulate_portfolio_history(
                start_date, end_date, exclude_symbols=["AAPL"]
            )

        self.assertEqual(len(result), 1)

        # The temporary portfolio should have dropped both AAPL transactions (txn1, txn3).
        _, kwargs = mock_ph.call_args
        temp_portfolio = kwargs["portfolio"]
        symbols = {t.instrument.symbol for t in temp_portfolio.transactions}
        self.assertNotIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)
        self.assertEqual(len(temp_portfolio.transactions), 1)

        # Original portfolio unchanged.
        self.assertEqual(len(self.portfolio_manager.current_portfolio.transactions), 3)

    def test_simulate_exclude_transaction_ids(self):
        """Test simulation with transaction ID exclusions."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        with patch("src.portfolio.portfolio_history.PortfolioHistory") as mock_ph:
            mock_ph.return_value.get_value_history.return_value = _make_history_df([11000.00])
            result = self.portfolio_manager.simulate_portfolio_history(
                start_date, end_date, exclude_transaction_ids=["txn2"]  # Exclude MSFT purchase
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result["total_value"].iloc[0], 11000.00)

        # txn2 excluded, txn1 and txn3 remain.
        _, kwargs = mock_ph.call_args
        temp_portfolio = kwargs["portfolio"]
        ids = {t.id for t in temp_portfolio.transactions}
        self.assertNotIn("txn2", ids)
        self.assertEqual(ids, {"txn1", "txn3"})

    def test_simulate_combined_exclusions(self):
        """Test simulation with both symbol and transaction exclusions."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        with patch("src.portfolio.portfolio_history.PortfolioHistory") as mock_ph:
            mock_ph.return_value.get_value_history.return_value = _make_history_df([10000.00])
            result = self.portfolio_manager.simulate_portfolio_history(
                start_date, end_date,
                exclude_symbols=["AAPL"],
                exclude_transaction_ids=["txn2"]
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result["total_value"].iloc[0], 10000.00)

        # Excluding AAPL (txn1, txn3) and txn2 leaves no transactions.
        _, kwargs = mock_ph.call_args
        temp_portfolio = kwargs["portfolio"]
        self.assertEqual(len(temp_portfolio.transactions), 0)

    def test_simulation_transaction_filtering(self):
        """Test that transaction filtering works correctly and the original is preserved."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)

        with patch("src.portfolio.portfolio_history.PortfolioHistory") as mock_ph:
            mock_ph.return_value.get_value_history.return_value = _make_history_df([])
            self.portfolio_manager.simulate_portfolio_history(
                start_date, end_date, exclude_symbols=["AAPL"]
            )

        # A temporary (filtered) portfolio must have been built with fewer transactions.
        _, kwargs = mock_ph.call_args
        temp_portfolio = kwargs["portfolio"]
        self.assertLess(len(temp_portfolio.transactions), len(self.test_portfolio.transactions))
        self.assertEqual(len(temp_portfolio.transactions), 1)  # only MSFT (txn2) remains

        # It must be a distinct object, not the live portfolio.
        self.assertIsNot(temp_portfolio, self.portfolio_manager.current_portfolio)

        # Verify original portfolio is still intact.
        self.assertEqual(len(self.portfolio_manager.current_portfolio.transactions), 3)

    def test_no_portfolio_error(self):
        """Test that simulation raises error when no portfolio is loaded."""
        self.portfolio_manager.current_portfolio = None

        with self.assertRaises(ValueError) as context:
            self.portfolio_manager.simulate_portfolio_history(
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
        # simulate_portfolio_history is called twice: baseline (no exclusions) then what-if.
        baseline_df = _make_history_df([10000, 12000])
        whatif_df = _make_history_df([10000, 11000])

        self.mock_portfolio_manager.simulate_portfolio_history.side_effect = [
            baseline_df,  # First call (baseline)
            whatif_df     # Second call (what-if)
        ]

        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31",
            exclude_symbols="AAPL"
        )

        self.assertIn("What-if Simulation", result)
        self.assertIn("AAPL", result)
        self.assertIn("12,000.00", result)  # Baseline end value
        self.assertIn("11,000.00", result)  # What-if end value
        self.assertIn("-1,000.00", result)  # Delta vs baseline (11000 - 12000)

    def test_what_if_tool_exclude_transactions(self):
        """Test what-if tool with transaction exclusions."""
        baseline_df = _make_history_df([15000, 15000])
        whatif_df = _make_history_df([13000, 13000])

        self.mock_portfolio_manager.simulate_portfolio_history.side_effect = [
            baseline_df,
            whatif_df
        ]

        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31",
            exclude_txn_ids="txn123,txn456"
        )

        self.assertIn("What-if Simulation", result)
        self.assertIn("txn123", result)  # Excluded transaction ids echoed back

    def test_what_if_tool_error_handling(self):
        """Test what-if tool error handling."""
        self.mock_portfolio_manager.simulate_portfolio_history.side_effect = Exception("Simulation failed")

        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31"
        )

        self.assertIn("Error running simulation", result)
        self.assertIn("Simulation failed", result)

    def test_what_if_tool_no_snapshots(self):
        """Test what-if tool when no data is generated."""
        self.mock_portfolio_manager.simulate_portfolio_history.return_value = pd.DataFrame()

        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31"
        )

        self.assertIn("No data generated", result)

    def test_what_if_tool_parameter_parsing(self):
        """Test parameter parsing in what-if tool."""
        baseline_df = _make_history_df([10000, 10000])
        whatif_df = _make_history_df([9000, 9000])

        self.mock_portfolio_manager.simulate_portfolio_history.side_effect = [
            baseline_df,
            whatif_df
        ]

        # Test with comma-separated symbols and transaction IDs
        result = self.tool._run(
            start="2024-01-01",
            end="2024-03-31",
            exclude_symbols="AAPL, MSFT, GOOGL",
            exclude_txn_ids="txn1, txn2"
        )

        self.assertIn("What-if Simulation", result)

        # Should properly parse and exclude multiple symbols/transactions
        calls = self.mock_portfolio_manager.simulate_portfolio_history.call_args_list
        self.assertEqual(len(calls), 2)

        # Check what-if call (second call) has exclusions
        whatif_call = calls[1]
        args, kwargs = whatif_call
        self.assertIn("AAPL", kwargs.get("exclude_symbols", []))
        self.assertIn("MSFT", kwargs.get("exclude_symbols", []))
        self.assertIn("GOOGL", kwargs.get("exclude_symbols", []))
        self.assertIn("txn1", kwargs.get("exclude_transaction_ids", []))
        self.assertIn("txn2", kwargs.get("exclude_transaction_ids", []))


if __name__ == '__main__':
    unittest.main()
