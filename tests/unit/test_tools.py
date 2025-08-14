"""
Unit tests for portfolio tools.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.agents.tools import (
    AddTransactionTool,
    CalculatorTool,
    GetCurrentPriceTool,
    GetPortfolioMetricsTool,
    GetPortfolioSummaryTool,
    GetTransactionHistoryTool,
    GetTransactionsTool,
    IngestPdfTool,
    SearchInstrumentTool,
    SimulateWhatIfTool,
    create_portfolio_tools,
)
from src.data_providers.base import InstrumentInfo
from src.portfolio.models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    Portfolio,
    Transaction,
    TransactionType,
)


class TestAddTransactionTool:
    """Test the AddTransactionTool."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        manager = Mock()
        manager.add_transaction.return_value = True
        manager.current_portfolio = Mock()
        return manager

    @pytest.fixture
    def tool(self, mock_portfolio_manager):
        """Create the tool instance."""
        return AddTransactionTool(mock_portfolio_manager)

    def test_add_buy_transaction(self, tool, mock_portfolio_manager):
        """Test adding a buy transaction."""
        result = tool._run(
            symbol="AAPL", transaction_type="buy", quantity=100, price=150.00, fees=1.00
        )

        assert "✅ Added buy transaction" in result
        mock_portfolio_manager.add_transaction.assert_called_once()
        call_args = mock_portfolio_manager.add_transaction.call_args
        assert call_args[1]["symbol"] == "AAPL"
        assert call_args[1]["transaction_type"] == TransactionType.BUY
        assert call_args[1]["quantity"] == Decimal("100")
        assert call_args[1]["price"] == Decimal("150.00")

    def test_add_sell_transaction(self, tool, mock_portfolio_manager):
        """Test adding a sell transaction."""
        result = tool._run(
            symbol="TSLA", transaction_type="sell", quantity=50, price=200.00, fees=1.00
        )

        assert "✅ Added sell transaction" in result
        mock_portfolio_manager.add_transaction.assert_called_once()

    def test_add_dividend_transaction(self, tool, mock_portfolio_manager):
        """Test adding a dividend transaction."""
        result = tool._run(
            symbol="AAPL", transaction_type="dividend", quantity=1, price=0.50, fees=0
        )

        assert "✅ Added dividend transaction" in result
        mock_portfolio_manager.add_transaction.assert_called_once()

    def test_add_cash_deposit(self, tool, mock_portfolio_manager):
        """Test adding a cash deposit."""
        result = tool._run(
            symbol="CASH", transaction_type="deposit", quantity=5000, price=0, fees=0
        )

        assert "✅ Added deposit transaction" in result
        call_args = mock_portfolio_manager.add_transaction.call_args
        assert call_args[1]["quantity"] == Decimal("1.0")
        assert call_args[1]["price"] == Decimal("5000")

    def test_add_cash_withdrawal(self, tool, mock_portfolio_manager):
        """Test adding a cash withdrawal."""
        result = tool._run(
            symbol="CASH", transaction_type="withdrawal", quantity=1000, price=0, fees=0
        )

        assert "✅ Added withdrawal transaction" in result

    def test_invalid_transaction_type(self, tool):
        """Test handling invalid transaction type."""
        result = tool._run(
            symbol="AAPL", transaction_type="invalid", quantity=100, price=150.00
        )

        assert "Invalid transaction type" in result

    def test_missing_symbol_and_isin(self, tool):
        """Test handling missing symbol and ISIN."""
        result = tool._run(transaction_type="buy", quantity=100, price=150.00)

        assert "Please provide either a symbol or an ISIN" in result

    def test_invalid_date_format(self, tool):
        """Test handling invalid date format."""
        result = tool._run(
            symbol="AAPL",
            transaction_type="buy",
            quantity=100,
            price=150.00,
            date="invalid-date",
        )

        assert "Invalid date format" in result

    def test_invalid_currency(self, tool):
        """Test handling invalid currency."""
        result = tool._run(
            symbol="AAPL",
            transaction_type="buy",
            quantity=100,
            price=150.00,
            currency="INVALID",
        )

        assert "Invalid currency" in result

    def test_transaction_failure(self, tool, mock_portfolio_manager):
        """Test handling transaction failure."""
        mock_portfolio_manager.add_transaction.return_value = False

        result = tool._run(
            symbol="AAPL", transaction_type="buy", quantity=100, price=150.00
        )

        assert "❌ Failed to add transaction" in result

    def test_use_isin_instead_of_symbol(self, tool, mock_portfolio_manager):
        """Test using ISIN instead of symbol."""
        result = tool._run(
            isin="US0378331005", transaction_type="buy", quantity=100, price=150.00
        )

        assert "✅ Added buy transaction" in result
        call_args = mock_portfolio_manager.add_transaction.call_args
        assert call_args[1]["isin"] == "US0378331005"

    def test_days_ago_parameter(self, tool, mock_portfolio_manager):
        """Test using days_ago parameter."""
        with patch("src.agents.tools.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.strptime = datetime.strptime

            result = tool._run(
                symbol="AAPL",
                transaction_type="buy",
                quantity=100,
                price=150.00,
                days_ago=2,
            )

            assert "✅ Added buy transaction" in result


class TestGetPortfolioSummaryTool:
    """Test the GetPortfolioSummaryTool."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        manager = Mock()
        portfolio = Mock()
        portfolio.name = "Test Portfolio"
        portfolio.created_at = datetime(2024, 1, 1, 10, 0, 0)
        portfolio.base_currency = Currency.USD
        portfolio.cash_balances = {Currency.USD: Decimal("1000.00")}
        manager.current_portfolio = portfolio
        manager.get_position_summary.return_value = [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "quantity": 100,
                "current_price": 150.00,
                "unrealized_pnl": 500.00,
                "unrealized_pnl_percent": 3.33,
            }
        ]
        manager.get_portfolio_value.return_value = 16000.00
        return manager

    @pytest.fixture
    def mock_metrics_calculator(self):
        """Create a mock metrics calculator."""
        calculator = Mock()
        return calculator

    @pytest.fixture
    def tool(self, mock_portfolio_manager, mock_metrics_calculator):
        """Create the tool instance."""
        return GetPortfolioSummaryTool(mock_portfolio_manager, mock_metrics_calculator)

    def test_get_portfolio_summary_with_metrics(self, tool, mock_portfolio_manager):
        """Test getting portfolio summary with metrics."""
        mock_portfolio_manager.get_performance_metrics.return_value = {
            "total_return_percent": 15.5,
            "annualized_volatility_percent": 12.3,
        }

        result = tool._run(include_metrics=True)

        assert "📊 **Portfolio Summary: Test Portfolio**" in result
        assert "💰 **Total Value**: $16,000.00 USD" in result
        assert "💵 **Cash Balances:**" in result
        assert "USD: 1,000.00" in result
        assert "📈 **Current Positions:**" in result
        assert "**AAPL** (Apple Inc.)" in result
        assert "📊 **Performance Metrics:**" in result
        assert "Total Return: 15.50%" in result

    def test_get_portfolio_summary_without_metrics(self, tool):
        """Test getting portfolio summary without metrics."""
        result = tool._run(include_metrics=False)

        assert "📊 **Portfolio Summary: Test Portfolio**" in result
        assert "📊 Performance Metrics:" not in result

    def test_no_portfolio_loaded(self, tool, mock_portfolio_manager):
        """Test handling when no portfolio is loaded."""
        mock_portfolio_manager.current_portfolio = None

        result = tool._run()

        assert "❌ No portfolio loaded" in result

    def test_empty_positions(self, tool, mock_portfolio_manager):
        """Test handling empty positions."""
        mock_portfolio_manager.get_position_summary.return_value = []

        result = tool._run()

        assert "📊 **Portfolio Summary: Test Portfolio**" in result
        assert "📈 Current Positions:" not in result

    def test_empty_cash_balances(self, tool, mock_portfolio_manager):
        """Test handling empty cash balances."""
        mock_portfolio_manager.current_portfolio.cash_balances = {}

        result = tool._run()

        assert "📊 **Portfolio Summary: Test Portfolio**" in result
        assert "💵 Cash Balances:" not in result


class TestGetTransactionsTool:
    """Test the GetTransactionsTool."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        manager = Mock()
        portfolio = Mock()

        # Create mock transactions
        instrument1 = Mock()
        instrument1.symbol = "AAPL"
        instrument1.currency = Currency.USD

        instrument2 = Mock()
        instrument2.symbol = "TSLA"
        instrument2.currency = Currency.USD

        txn1 = Mock()
        txn1.id = "txn1"
        txn1.timestamp = datetime(2024, 1, 15, 10, 30, 0)
        txn1.instrument = instrument1
        txn1.transaction_type = TransactionType.BUY
        txn1.quantity = Decimal("100")
        txn1.price = Decimal("150.00")
        txn1.fees = Decimal("1.00")
        txn1.currency = Currency.USD
        txn1.notes = "Test transaction"

        txn2 = Mock()
        txn2.id = "txn2"
        txn2.timestamp = datetime(2024, 1, 20, 14, 15, 0)
        txn2.instrument = instrument2
        txn2.transaction_type = TransactionType.SELL
        txn2.quantity = Decimal("50")
        txn2.price = Decimal("200.00")
        txn2.fees = Decimal("1.00")
        txn2.currency = Currency.USD
        txn2.notes = None

        portfolio.transactions = [txn1, txn2]
        manager.current_portfolio = portfolio
        return manager

    @pytest.fixture
    def tool(self, mock_portfolio_manager):
        """Create the tool instance."""
        return GetTransactionsTool(mock_portfolio_manager)

    def test_get_transactions(self, tool):
        """Test getting transactions."""
        result = tool._run()

        assert "🧾 Transactions (all):" in result
        assert "id,timestamp,symbol,type,quantity,price,fees,currency,notes" in result
        assert (
            "txn1,2024-01-15T10:30:00,AAPL,buy,100,150.00,1.00,USD,Test transaction"
            in result
        )
        assert "txn2,2024-01-20T14:15:00,TSLA,sell,50,200.00,1.00,USD," in result

    def test_no_portfolio_loaded(self, tool, mock_portfolio_manager):
        """Test handling when no portfolio is loaded."""
        mock_portfolio_manager.current_portfolio = None

        result = tool._run()

        assert "❌ No portfolio loaded" in result

    def test_empty_transactions(self, tool, mock_portfolio_manager):
        """Test handling empty transactions."""
        mock_portfolio_manager.current_portfolio.transactions = []

        result = tool._run()

        assert "🧾 Transactions (all):" in result
        assert "id,timestamp,symbol,type,quantity,price,fees,currency,notes" in result
        # Only header should be present


class TestSimulateWhatIfTool:
    """Test the SimulateWhatIfTool."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        manager = Mock()
        portfolio = Mock()
        manager.current_portfolio = portfolio

        # Mock snapshots
        baseline_snaps = [
            Mock(total_value=Decimal("10000.00")),
            Mock(total_value=Decimal("11000.00")),
        ]

        whatif_snaps = [
            Mock(total_value=Decimal("10000.00")),
            Mock(total_value=Decimal("10500.00")),
        ]

        manager.simulate_snapshots_for_range.side_effect = [
            baseline_snaps,  # First call (baseline)
            whatif_snaps,  # Second call (what-if)
        ]

        return manager

    @pytest.fixture
    def tool(self, mock_portfolio_manager):
        """Create the tool instance."""
        return SimulateWhatIfTool(mock_portfolio_manager)

    def test_simulate_what_if(self, tool):
        """Test simulating what-if scenario."""
        result = tool._run(
            start="2024-01-01",
            end="2024-01-31",
            exclude_symbols="AAPL,TSLA",
            exclude_txn_ids="txn1,txn2",
        )

        assert "📈 What-if Simulation (2024-01-01 → 2024-01-31)" in result
        assert "Excluded symbols: ['AAPL', 'TSLA']" in result
        assert "Excluded txn ids: ['txn1', 'txn2']" in result
        assert "Start value: $10,000.00" in result
        assert "End value: $10,500.00" in result
        assert "Total return: 5.00%" in result

    def test_no_portfolio_loaded(self, tool, mock_portfolio_manager):
        """Test handling when no portfolio is loaded."""
        mock_portfolio_manager.current_portfolio = None

        result = tool._run(start="2024-01-01", end="2024-01-31")

        assert "❌ No portfolio loaded" in result

    def test_invalid_date_range(self, tool):
        """Test handling invalid date range."""
        result = tool._run(start="2024-01-31", end="2024-01-01")

        assert "❌ Start date must be on or before end date" in result

    def test_no_snapshots_generated(self, tool, mock_portfolio_manager):
        """Test handling when no snapshots are generated."""
        # Mock to return empty for both baseline and what-if scenarios
        mock_portfolio_manager.simulate_snapshots_for_range.side_effect = [[], []]

        result = tool._run(start="2024-01-01", end="2024-01-31")

        assert "No snapshots generated for the specified range" in result


class TestSearchInstrumentTool:
    """Test the SearchInstrumentTool."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        manager = Mock()
        return manager

    @pytest.fixture
    def tool(self, mock_data_manager):
        """Create the tool instance."""
        return SearchInstrumentTool(mock_data_manager)

    def test_search_instruments(self, tool, mock_data_manager):
        """Test searching instruments."""
        mock_results = [
            InstrumentInfo(
                symbol="AAPL",
                name="Apple Inc.",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
                exchange="NASDAQ",
            ),
            InstrumentInfo(
                symbol="TSLA",
                name="Tesla Inc.",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
                exchange="NASDAQ",
            ),
        ]
        mock_data_manager.search_instruments.return_value = mock_results

        result = tool._run("Apple")

        assert "🔍 **Search Results for 'Apple':**" in result
        assert "**AAPL** - Apple Inc." in result
        assert "**TSLA** - Tesla Inc." in result
        assert "Type: Stock" in result
        assert "Currency: USD" in result

    def test_no_instruments_found(self, tool, mock_data_manager):
        """Test handling when no instruments are found."""
        mock_data_manager.search_instruments.return_value = []

        result = tool._run("InvalidSymbol")

        assert "❌ No instruments found for 'InvalidSymbol'" in result

    def test_search_error(self, tool, mock_data_manager):
        """Test handling search errors."""
        mock_data_manager.search_instruments.side_effect = Exception("API Error")

        result = tool._run("AAPL")

        assert "❌ Error searching instruments" in result

    def test_limit_results_to_10(self, tool, mock_data_manager):
        """Test that results are limited to 10."""
        # Create 15 mock results
        mock_results = []
        for i in range(15):
            mock_results.append(
                InstrumentInfo(
                    symbol=f"SYMBOL{i}",
                    name=f"Company {i}",
                    instrument_type=InstrumentType.STOCK,
                    currency=Currency.USD,
                )
            )

        mock_data_manager.search_instruments.return_value = mock_results

        result = tool._run("test")

        # Should only show 10 results
        assert result.count("• **SYMBOL") == 10


class TestGetCurrentPriceTool:
    """Test the GetCurrentPriceTool."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        manager = Mock()
        return manager

    @pytest.fixture
    def tool(self, mock_data_manager):
        """Create the tool instance."""
        return GetCurrentPriceTool(mock_data_manager)

    def test_get_current_price(self, tool, mock_data_manager):
        """Test getting current price."""
        mock_data_manager.get_current_price.return_value = Decimal("150.00")
        mock_data_manager.get_instrument_info.return_value = InstrumentInfo(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )

        result = tool._run("AAPL")

        assert "💰 **AAPL** (Apple Inc.): $150.00" in result

    def test_price_not_found(self, tool, mock_data_manager):
        """Test handling when price is not found."""
        mock_data_manager.get_current_price.return_value = None

        result = tool._run("INVALID")

        assert "❌ Could not get current price for INVALID" in result

    def test_instrument_info_not_found(self, tool, mock_data_manager):
        """Test handling when instrument info is not found."""
        mock_data_manager.get_current_price.return_value = Decimal("150.00")
        mock_data_manager.get_instrument_info.return_value = None

        result = tool._run("AAPL")

        assert "💰 **AAPL** (AAPL): $150.00" in result  # Uses symbol as fallback

    def test_get_price_error(self, tool, mock_data_manager):
        """Test handling get price errors."""
        mock_data_manager.get_current_price.side_effect = Exception("API Error")

        result = tool._run("AAPL")

        assert "❌ Error getting price for AAPL" in result


class TestGetPortfolioMetricsTool:
    """Test the GetPortfolioMetricsTool."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        manager = Mock()
        portfolio = Mock()
        manager.current_portfolio = portfolio
        return manager

    @pytest.fixture
    def mock_metrics_calculator(self):
        """Create a mock metrics calculator."""
        calculator = Mock()
        return calculator

    @pytest.fixture
    def tool(self, mock_portfolio_manager, mock_metrics_calculator):
        """Create the tool instance."""
        return GetPortfolioMetricsTool(mock_portfolio_manager, mock_metrics_calculator)

    def test_get_portfolio_metrics(
        self, tool, mock_portfolio_manager, mock_metrics_calculator
    ):
        """Test getting portfolio metrics."""
        # Mock snapshots
        mock_snapshots = [Mock(), Mock()]
        mock_portfolio_manager.storage.load_snapshots.return_value = mock_snapshots

        # Mock metrics
        mock_metrics = {
            "total_return": 0.15,
            "annualized_return": 0.12,
            "volatility": 0.18,
            "max_drawdown": 0.08,
            "var_5pct": 0.05,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "calmar_ratio": 1.8,
            "benchmark_available": True,
            "beta": 0.95,
            "alpha": 0.02,
            "information_ratio": 0.8,
            "benchmark_return": 0.13,
        }
        mock_metrics_calculator.calculate_portfolio_metrics.return_value = mock_metrics

        result = tool._run(days=365, benchmark="SPY")

        assert "📊 **Portfolio Metrics** (365 days vs SPY)" in result
        assert "Total Return: 15.00%" in result
        assert "Annualized Return: 12.00%" in result
        assert "Volatility: 18.00%" in result
        assert "Max Drawdown: 8.00%" in result
        assert "Sharpe Ratio: 1.200" in result
        assert "Beta: 0.950" in result
        assert "Alpha: 2.00%" in result

    def test_no_portfolio_loaded(self, tool, mock_portfolio_manager):
        """Test handling when no portfolio is loaded."""
        mock_portfolio_manager.current_portfolio = None

        result = tool._run()

        assert "❌ No portfolio loaded" in result

    def test_insufficient_historical_data(self, tool, mock_portfolio_manager):
        """Test handling insufficient historical data."""
        mock_portfolio_manager.storage.load_snapshots.return_value = [Mock()]

        result = tool._run()

        assert "❌ Insufficient historical data for metrics calculation" in result

    def test_metrics_calculation_error(
        self, tool, mock_portfolio_manager, mock_metrics_calculator
    ):
        """Test handling metrics calculation errors."""
        mock_snapshots = [Mock(), Mock()]
        mock_portfolio_manager.storage.load_snapshots.return_value = mock_snapshots
        mock_metrics_calculator.calculate_portfolio_metrics.return_value = {
            "error": "Calculation failed"
        }

        result = tool._run()

        assert "❌ Calculation failed" in result

    def test_metrics_without_benchmark(
        self, tool, mock_portfolio_manager, mock_metrics_calculator
    ):
        """Test metrics without benchmark comparison."""
        mock_snapshots = [Mock(), Mock()]
        mock_portfolio_manager.storage.load_snapshots.return_value = mock_snapshots

        mock_metrics = {
            "total_return": 0.15,
            "annualized_return": 0.12,
            "volatility": 0.18,
            "max_drawdown": 0.08,
            "var_5pct": 0.05,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "calmar_ratio": 1.8,
            "benchmark_available": False,
        }
        mock_metrics_calculator.calculate_portfolio_metrics.return_value = mock_metrics

        result = tool._run()

        assert "📊 **Portfolio Metrics** (365 days vs SPY)" in result
        assert "Total Return: 15.00%" in result
        assert "vs SPY:" not in result  # Should not show benchmark comparison


class TestGetTransactionHistoryTool:
    """Test the GetTransactionHistoryTool."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        manager = Mock()
        portfolio = Mock()
        manager.current_portfolio = portfolio
        return manager

    @pytest.fixture
    def tool(self, mock_portfolio_manager):
        """Create the tool instance."""
        return GetTransactionHistoryTool(mock_portfolio_manager)

    def test_get_transaction_history(self, tool, mock_portfolio_manager):
        """Test getting transaction history."""
        mock_transactions = [
            {
                "timestamp": datetime(2024, 1, 15, 10, 30, 0),
                "type": TransactionType.BUY,
                "symbol": "AAPL",
                "quantity": 100,
                "price": 150.00,
                "total_value": 15000.00,
            },
            {
                "timestamp": datetime(2024, 1, 20, 14, 15, 0),
                "type": TransactionType.SELL,
                "symbol": "TSLA",
                "quantity": 50,
                "price": 200.00,
                "total_value": 10000.00,
            },
        ]
        mock_portfolio_manager.get_transaction_history.return_value = mock_transactions

        result = tool._run(days=30)

        assert "📝 **Transaction History** (Last 30 days)" in result
        assert "2024-01-15: BUY 100 AAPL @ $150.0 (Total: $15000.0)" in result
        assert "2024-01-20: SELL 50 TSLA @ $200.0 (Total: $10000.0)" in result

    def test_no_portfolio_loaded(self, tool, mock_portfolio_manager):
        """Test handling when no portfolio is loaded."""
        mock_portfolio_manager.current_portfolio = None

        result = tool._run()

        assert "❌ No portfolio loaded" in result

    def test_no_transactions_found(self, tool, mock_portfolio_manager):
        """Test handling when no transactions are found."""
        mock_portfolio_manager.get_transaction_history.return_value = []

        result = tool._run(days=30)

        assert "📝 No transactions found in the last 30 days" in result

    def test_limit_to_20_transactions(self, tool, mock_portfolio_manager):
        """Test that results are limited to 20 transactions."""
        # Create 25 mock transactions
        mock_transactions = []
        for i in range(25):
            mock_transactions.append(
                {
                    "timestamp": datetime(2024, 1, i + 1, 10, 0, 0),
                    "type": TransactionType.BUY,
                    "symbol": f"SYMBOL{i}",
                    "quantity": 100,
                    "price": 100.00,
                    "total_value": 10000.00,
                }
            )

        mock_portfolio_manager.get_transaction_history.return_value = mock_transactions

        result = tool._run(days=30)

        # Should only show 20 transactions
        assert result.count("• 2024-01-") == 20


class TestCalculatorTool:
    """Test the CalculatorTool."""

    @pytest.fixture
    def tool(self):
        """Create the tool instance."""
        return CalculatorTool()

    def test_basic_arithmetic(self, tool):
        """Test basic arithmetic operations."""
        result = tool._run("2 + 3")
        assert "🧮 2 + 3 = 5" in result

        result = tool._run("10 - 4")
        assert "🧮 10 - 4 = 6" in result

        result = tool._run("6 * 7")
        assert "🧮 6 * 7 = 42" in result

        result = tool._run("15 / 3")
        assert "🧮 15 / 3 = 5.0" in result

    def test_complex_expressions(self, tool):
        """Test complex mathematical expressions."""
        result = tool._run("2 * (3 + 4)")
        assert "🧮 2 * (3 + 4) = 14" in result

        result = tool._run("10 ** 2")
        assert "🧮 10 ** 2 = 100" in result

        result = tool._run("17 % 5")
        assert "🧮 17 % 5 = 2" in result

    def test_math_functions(self, tool):
        """Test mathematical functions."""
        result = tool._run("sin(pi/2)")
        assert "🧮 sin(pi/2) = 1.0" in result

        result = tool._run("log(100, 10)")
        assert "🧮 log(100, 10) = 2.0" in result

        result = tool._run("sqrt(16)")
        assert "🧮 sqrt(16) = 4.0" in result

    def test_invalid_expression(self, tool):
        """Test handling invalid expressions."""
        result = tool._run("invalid expression")
        assert "❌ Error evaluating expression" in result

        result = tool._run("1 / 0")
        assert "❌ Error evaluating expression" in result


class TestIngestPdfTool:
    """Test the IngestPdfTool."""

    @pytest.fixture
    def tool(self):
        """Create the tool instance."""
        return IngestPdfTool()

    @patch("src.agents.tools.PdfReader")
    def test_ingest_pdf_success(self, mock_pdf_reader, tool):
        """Test successful PDF ingestion."""
        # Mock PDF pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"

        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_reader = Mock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader

        result = tool._run("/path/to/test.pdf")

        assert "📄 PDF Content Extracted (length: 30 chars)" in result
        assert "Page 1 content" in result
        assert "Page 2 content" in result

    @patch("src.agents.tools.PdfReader")
    def test_ingest_pdf_no_text(self, mock_pdf_reader, tool):
        """Test PDF with no extractable text."""
        mock_page = Mock()
        mock_page.extract_text.return_value = ""

        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        result = tool._run("/path/to/test.pdf")

        assert "❌ No text extracted from PDF" in result

    @patch("src.agents.tools.PdfReader")
    def test_ingest_pdf_page_error(self, mock_pdf_reader, tool):
        """Test handling page extraction errors."""
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"

        mock_page2 = Mock()
        mock_page2.extract_text.side_effect = Exception("Page error")

        mock_reader = Mock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader

        result = tool._run("/path/to/test.pdf")

        assert "📄 PDF Content Extracted (length: 14 chars)" in result
        assert "Page 1 content" in result

    def test_file_not_found(self, tool):
        """Test handling file not found."""
        result = tool._run("/nonexistent/file.pdf")

        assert "❌ File not found: /nonexistent/file.pdf" in result

    @patch("src.agents.tools.PdfReader")
    def test_ingest_pdf_general_error(self, mock_pdf_reader, tool):
        """Test handling general PDF reading errors."""
        mock_pdf_reader.side_effect = Exception("PDF reading error")

        result = tool._run("/path/to/test.pdf")

        assert "❌ Error reading PDF" in result

    @patch("src.agents.tools.PdfReader")
    def test_ingest_pdf_content_truncation(self, mock_pdf_reader, tool):
        """Test PDF content truncation for very long content."""
        # Create very long content
        long_content = "A" * 250000

        mock_page = Mock()
        mock_page.extract_text.return_value = long_content

        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        result = tool._run("/path/to/test.pdf")

        assert "📄 PDF Content Extracted (length: 200016 chars)" in result
        assert "...[truncated]" in result


class TestCreatePortfolioTools:
    """Test the create_portfolio_tools function."""

    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create a mock portfolio manager."""
        return Mock()

    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager."""
        return Mock()

    @pytest.fixture
    def mock_metrics_calculator(self):
        """Create a mock metrics calculator."""
        return Mock()

    def test_create_portfolio_tools(
        self, mock_portfolio_manager, mock_data_manager, mock_metrics_calculator
    ):
        """Test creating all portfolio tools."""
        tools = create_portfolio_tools(
            mock_portfolio_manager, mock_data_manager, mock_metrics_calculator
        )

        # Check that all expected tools are created
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "add_transaction",
            "get_portfolio_summary",
            "get_transactions",
            "simulate_what_if",
            "ingest_pdf",
            "calculator",
            "search_instrument",
            "get_current_price",
            "get_portfolio_metrics",
            "get_transaction_history",
        ]

        assert len(tools) == len(expected_tools)
        for expected_tool in expected_tools:
            assert expected_tool in tool_names

    def test_tool_initialization(
        self, mock_portfolio_manager, mock_data_manager, mock_metrics_calculator
    ):
        """Test that tools are properly initialized with dependencies."""
        tools = create_portfolio_tools(
            mock_portfolio_manager, mock_data_manager, mock_metrics_calculator
        )

        # Check that tools have the correct dependencies
        add_transaction_tool = next(t for t in tools if t.name == "add_transaction")
        assert add_transaction_tool.portfolio_manager == mock_portfolio_manager

        get_portfolio_summary_tool = next(
            t for t in tools if t.name == "get_portfolio_summary"
        )
        assert get_portfolio_summary_tool.portfolio_manager == mock_portfolio_manager
        assert get_portfolio_summary_tool.metrics_calculator == mock_metrics_calculator

        search_instrument_tool = next(t for t in tools if t.name == "search_instrument")
        assert search_instrument_tool.data_manager == mock_data_manager
