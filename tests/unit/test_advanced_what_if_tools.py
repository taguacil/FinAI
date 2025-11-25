"""
Unit tests for advanced what-if analysis tools.
Tests the enhanced AI agent capabilities for portfolio scenario modeling.
"""

import unittest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import Mock, patch

from src.portfolio.models import (
    Portfolio, Transaction, TransactionType, FinancialInstrument,
    Currency, InstrumentType, PortfolioSnapshot, Position
)
from src.portfolio.manager import PortfolioManager
from src.portfolio.storage import FileBasedStorage
from src.agents.tools import AdvancedWhatIfTool, HypotheticalPositionTool


class TestAdvancedWhatIfTools(unittest.TestCase):
    """Test advanced what-if analysis tools for the AI agent."""

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
                quantity=Decimal("100"),
                price=Decimal("150.00"),
                currency=Currency.USD,
                notes="Buy AAPL"
            ),
            Transaction(
                id="txn2",
                timestamp=datetime.combine(base_date, datetime.min.time()),
                instrument=self.msft_instrument,
                transaction_type=TransactionType.BUY,
                quantity=Decimal("50"),
                price=Decimal("300.00"),
                currency=Currency.USD,
                notes="Buy MSFT"
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

        # Create test tools
        self.advanced_tool = AdvancedWhatIfTool(self.portfolio_manager)
        self.hypothetical_tool = HypotheticalPositionTool(self.portfolio_manager)

    def test_advanced_what_if_tool_initialization(self):
        """Test advanced what-if tool initialization."""
        self.assertEqual(self.advanced_tool.name, "advanced_what_if")
        self.assertIn("portfolio modifications", self.advanced_tool.description.lower())
        self.assertEqual(self.advanced_tool.portfolio_manager, self.portfolio_manager)

    def test_hypothetical_position_tool_initialization(self):
        """Test hypothetical position tool initialization."""
        self.assertEqual(self.hypothetical_tool.name, "test_hypothetical_position")
        self.assertIn("hypothetical position", self.hypothetical_tool.description.lower())
        self.assertEqual(self.hypothetical_tool.portfolio_manager, self.portfolio_manager)

    def test_parse_position_modifications(self):
        """Test parsing of position modification strings."""
        # Test percentage changes
        modifications = self.advanced_tool._parse_position_modifications("AAPL:+50%,MSFT:-25%")
        self.assertEqual(modifications["AAPL"], {"type": "percent", "value": 50.0})
        self.assertEqual(modifications["MSFT"], {"type": "percent", "value": -25.0})

        # Test absolute values
        modifications = self.advanced_tool._parse_position_modifications("AAPL:=200,MSFT:=75")
        self.assertEqual(modifications["AAPL"], {"type": "absolute", "value": 200.0})
        self.assertEqual(modifications["MSFT"], {"type": "absolute", "value": 75.0})

        # Test delta changes
        modifications = self.advanced_tool._parse_position_modifications("AAPL:+100,MSFT:-25")
        self.assertEqual(modifications["AAPL"], {"type": "delta", "value": 100.0})
        self.assertEqual(modifications["MSFT"], {"type": "delta", "value": -25.0})

        # Test mixed format
        modifications = self.advanced_tool._parse_position_modifications("AAPL:+50%,MSFT:=100,GOOGL:-10")
        self.assertEqual(modifications["AAPL"], {"type": "percent", "value": 50.0})
        self.assertEqual(modifications["MSFT"], {"type": "absolute", "value": 100.0})
        self.assertEqual(modifications["GOOGL"], {"type": "delta", "value": -10.0})

    def test_parse_new_positions(self):
        """Test parsing of new position strings."""
        new_positions = self.advanced_tool._parse_new_positions("NVDA:100@$800,TSLA:50@$250")

        self.assertIn("NVDA", new_positions)
        self.assertEqual(new_positions["NVDA"]["quantity"], 100.0)
        self.assertEqual(new_positions["NVDA"]["price"], 800.0)

        self.assertIn("TSLA", new_positions)
        self.assertEqual(new_positions["TSLA"]["quantity"], 50.0)
        self.assertEqual(new_positions["TSLA"]["price"], 250.0)

        # Test with different price formats
        new_positions = self.advanced_tool._parse_new_positions("AMZN:25@3500,GOOGL:10@2800")
        self.assertEqual(new_positions["AMZN"]["price"], 3500.0)
        self.assertEqual(new_positions["GOOGL"]["price"], 2800.0)

    @patch('src.portfolio.scenarios.PortfolioScenarioEngine')
    def test_advanced_what_if_basic_execution(self, mock_engine_class):
        """Test basic execution of advanced what-if tool."""
        # Mock scenario engine and results
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        mock_result = Mock()
        mock_result.get_summary_stats.return_value = {
            'mean_final_value': 150000.0,
            'median_final_value': 145000.0,
            'std_final_value': 25000.0,
            'probability_of_loss': 0.15,
            'probability_of_doubling': 0.25,
            'mean_annualized_return': 0.08,
            'mean_sharpe_ratio': 1.2,
            'mean_max_drawdown': -0.12,
            'var_95': 110000.0
        }
        mock_result.scenario_config.name = "Custom Likely Scenario"
        mock_result.scenario_config.projection_years = 2.0
        mock_result.scenario_config.monte_carlo_runs = 1000
        mock_result.scenario_config.market_assumptions.expected_return = 0.08
        mock_result.scenario_config.market_assumptions.volatility = 0.20

        mock_engine.run_scenario_simulation.return_value = mock_result

        # Mock create_snapshot to return test data
        with patch.object(self.portfolio_manager, 'create_snapshot') as mock_create_snapshot:
            test_snapshot = PortfolioSnapshot(
                date=date.today(),
                total_value=Decimal("125000.00"),
                cash_balance=Decimal("10000.00"),
                positions_value=Decimal("115000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("10000.00")},
                total_cost_basis=Decimal("120000.00"),
                total_unrealized_pnl=Decimal("5000.00"),
                total_unrealized_pnl_percent=Decimal("4.17")
            )
            mock_create_snapshot.return_value = test_snapshot

            # Run the advanced what-if analysis
            result = self.advanced_tool._run(
                scenario_type="likely",
                projection_years=2.0,
                monte_carlo_runs=1000,
                modify_positions="AAPL:+50%",
                market_return=0.08,
                market_volatility=0.20
            )

            # Verify execution
            self.assertIn("Advanced What-If Analysis", result)
            self.assertIn("AAPL:+50%", result)
            self.assertIn("$150,000", result)
            self.assertIn("8.0%", result)  # Market return
            self.assertIn("20.0%", result)  # Volatility

            # Verify engine was called
            mock_engine.run_scenario_simulation.assert_called_once()

    @patch('src.portfolio.scenarios.PortfolioScenarioEngine')
    def test_advanced_what_if_with_new_positions(self, mock_engine_class):
        """Test advanced what-if with new position additions."""
        # Mock scenario engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        mock_result = Mock()
        mock_result.get_summary_stats.return_value = {
            'mean_final_value': 175000.0,
            'median_final_value': 170000.0,
            'std_final_value': 30000.0,
            'probability_of_loss': 0.10,
            'probability_of_doubling': 0.30,
            'mean_annualized_return': 0.12,
            'mean_sharpe_ratio': 1.5,
            'mean_max_drawdown': -0.08,
            'var_95': 130000.0
        }
        mock_result.scenario_config.name = "Custom Optimistic Scenario"
        mock_result.scenario_config.projection_years = 1.0
        mock_result.scenario_config.monte_carlo_runs = 500
        mock_result.scenario_config.market_assumptions.expected_return = 0.12
        mock_result.scenario_config.market_assumptions.volatility = 0.16

        mock_engine.run_scenario_simulation.return_value = mock_result

        # Mock create_snapshot
        with patch.object(self.portfolio_manager, 'create_snapshot') as mock_create_snapshot:
            test_snapshot = PortfolioSnapshot(
                date=date.today(),
                total_value=Decimal("125000.00"),
                cash_balance=Decimal("10000.00"),
                positions_value=Decimal("115000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("10000.00")},
                total_cost_basis=Decimal("120000.00"),
                total_unrealized_pnl=Decimal("5000.00"),
                total_unrealized_pnl_percent=Decimal("4.17")
            )
            mock_create_snapshot.return_value = test_snapshot

            # Run analysis with new positions
            result = self.advanced_tool._run(
                scenario_type="optimistic",
                projection_years=1.0,
                monte_carlo_runs=500,
                add_positions="NVDA:10@$800,TSLA:20@$250",
                market_return=0.12,
                market_volatility=0.16
            )

            # Verify results
            self.assertIn("Advanced What-If Analysis", result)
            self.assertIn("NVDA:10@$800,TSLA:20@$250", result)
            self.assertIn("$175,000", result)
            self.assertIn("12.0%", result)  # Market return

    @patch('src.agents.tools.AdvancedWhatIfTool')
    def test_hypothetical_position_tool(self, mock_advanced_tool_class):
        """Test hypothetical position tool functionality."""
        # Mock the advanced tool
        mock_advanced_tool = Mock()
        mock_advanced_tool_class.return_value = mock_advanced_tool

        mock_advanced_tool._run.return_value = """
        🔮 Advanced What-If Analysis
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        📊 Portfolio Modifications Applied:
           ➕ New Positions: AAPL:100@$150
           📈 Immediate Impact: +$15,000.00 (+12.0%)

        📈 Projection Results:
           • Mean Final Value: $165,000.00
        """

        # Mock create_snapshot
        with patch.object(self.portfolio_manager, 'create_snapshot') as mock_create_snapshot:
            test_snapshot = PortfolioSnapshot(
                date=date.today(),
                total_value=Decimal("125000.00"),
                cash_balance=Decimal("10000.00"),
                positions_value=Decimal("115000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("10000.00")},
                total_cost_basis=Decimal("120000.00"),
                total_unrealized_pnl=Decimal("5000.00"),
                total_unrealized_pnl_percent=Decimal("4.17")
            )
            mock_create_snapshot.return_value = test_snapshot

            # Test hypothetical position
            result = self.hypothetical_tool._run(
                symbol="AAPL",
                quantity=100,
                purchase_price=150.0,
                scenario="likely",
                time_horizon=1.0
            )

            # Verify execution
            self.assertIn("Hypothetical Position Analysis", result)
            self.assertIn("AAPL", result)
            self.assertIn("100 shares", result)
            self.assertIn("$150.00", result)

            # Verify advanced tool was called
            mock_advanced_tool._run.assert_called_once()

    def test_hypothetical_position_with_investment_amount(self):
        """Test hypothetical position tool with investment amount instead of quantity."""
        # Mock create_snapshot
        with patch.object(self.portfolio_manager, 'create_snapshot') as mock_create_snapshot:
            test_snapshot = PortfolioSnapshot(
                date=date.today(),
                total_value=Decimal("125000.00"),
                cash_balance=Decimal("10000.00"),
                positions_value=Decimal("115000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("10000.00")},
                total_cost_basis=Decimal("120000.00"),
                total_unrealized_pnl=Decimal("5000.00"),
                total_unrealized_pnl_percent=Decimal("4.17")
            )
            mock_create_snapshot.return_value = test_snapshot

            # Mock the advanced tool
            with patch.object(self.hypothetical_tool, 'portfolio_manager') as mock_pm:
                mock_pm.current_portfolio = self.test_portfolio

                # Test with investment amount
                with patch('src.agents.tools.AdvancedWhatIfTool') as mock_advanced_class:
                    mock_advanced_tool = Mock()
                    mock_advanced_class.return_value = mock_advanced_tool
                    mock_advanced_tool._run.return_value = "Mock result"

                    result = self.hypothetical_tool._run(
                        symbol="TSLA",
                        quantity=0,  # Should be ignored when investment_amount is provided
                        purchase_price=250.0,
                        investment_amount="$5000",
                        scenario="optimistic"
                    )

                    # Verify the call was made with correct calculated quantity
                    call_args = mock_advanced_tool._run.call_args
                    self.assertIn("add_positions", call_args[1])
                    # 5000 / 250 = 20 shares
                    self.assertIn("TSLA:20.0@$250.0", call_args[1]["add_positions"])

    def test_error_handling_no_portfolio(self):
        """Test error handling when no portfolio is loaded."""
        # Create tools with no portfolio
        empty_manager = PortfolioManager(
            storage=self.mock_storage,
            data_manager=self.mock_data_manager
        )
        empty_manager.current_portfolio = None

        advanced_tool = AdvancedWhatIfTool(empty_manager)
        hypothetical_tool = HypotheticalPositionTool(empty_manager)

        # Test advanced tool
        result = advanced_tool._run()
        self.assertIn("❌ No portfolio loaded", result)

        # Test hypothetical tool
        result = hypothetical_tool._run("AAPL", 100, 150.0)
        self.assertIn("❌ No portfolio loaded", result)

    def test_parameter_validation(self):
        """Test parameter validation in advanced what-if tool."""
        with patch.object(self.portfolio_manager, 'create_snapshot') as mock_create_snapshot:
            test_snapshot = PortfolioSnapshot(
                date=date.today(),
                total_value=Decimal("125000.00"),
                cash_balance=Decimal("10000.00"),
                positions_value=Decimal("115000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("10000.00")},
                total_cost_basis=Decimal("120000.00"),
                total_unrealized_pnl=Decimal("5000.00"),
                total_unrealized_pnl_percent=Decimal("4.17")
            )
            mock_create_snapshot.return_value = test_snapshot

            with patch('src.portfolio.scenarios.PortfolioScenarioEngine') as mock_engine_class:
                mock_engine = Mock()
                mock_engine_class.return_value = mock_engine
                mock_result = Mock()
                mock_result.get_summary_stats.return_value = {
                    'mean_final_value': 140000.0,
                    'median_final_value': 135000.0,
                    'std_final_value': 20000.0,
                    'probability_of_loss': 0.20,
                    'probability_of_doubling': 0.15,
                    'mean_annualized_return': 0.06,
                    'mean_sharpe_ratio': 1.0,
                    'mean_max_drawdown': -0.15,
                    'var_95': 100000.0
                }
                mock_result.scenario_config = Mock()
                mock_result.scenario_config.name = "Test"
                mock_result.scenario_config.projection_years = 2.0
                mock_result.scenario_config.monte_carlo_runs = 1000
                mock_result.scenario_config.market_assumptions = Mock()
                mock_result.scenario_config.market_assumptions.expected_return = 0.08
                mock_result.scenario_config.market_assumptions.volatility = 0.20
                mock_engine.run_scenario_simulation.return_value = mock_result

                # Test parameter clamping
                result = self.advanced_tool._run(
                    projection_years=15.0,  # Should be clamped to 10.0
                    monte_carlo_runs=10000,  # Should be clamped to 5000
                    market_return=2.0,  # Should be clamped to 1.0
                    market_volatility=-0.5  # Should be clamped to 0.01
                )

                # Verify the tool didn't crash and produced output
                self.assertIn("Advanced What-If Analysis", result)

    def test_stress_test_parameter_adjustment(self):
        """Test that stress test parameters are properly adjusted."""
        with patch.object(self.portfolio_manager, 'create_snapshot') as mock_create_snapshot:
            test_snapshot = PortfolioSnapshot(
                date=date.today(),
                total_value=Decimal("125000.00"),
                cash_balance=Decimal("10000.00"),
                positions_value=Decimal("115000.00"),
                base_currency=Currency.USD,
                positions={},
                cash_balances={Currency.USD: Decimal("10000.00")},
                total_cost_basis=Decimal("120000.00"),
                total_unrealized_pnl=Decimal("5000.00"),
                total_unrealized_pnl_percent=Decimal("4.17")
            )
            mock_create_snapshot.return_value = test_snapshot

            # Test stress test adjustments
            scenario_config = self.advanced_tool._create_scenario_config(
                "stress", 2.0, 1000, 0.05, 0.15, 0.0, True
            )

            # Stress test should force negative returns and high volatility
            self.assertLessEqual(scenario_config.market_assumptions.expected_return, -0.02)
            self.assertGreaterEqual(scenario_config.market_assumptions.volatility, 0.35)

            # Test stress correlations
            self.assertGreater(scenario_config.market_assumptions.equity_correlation, 0.8)


if __name__ == '__main__':
    unittest.main()
