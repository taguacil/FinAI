"""
Unit tests for the advanced scenario modeling engine.
Tests Monte Carlo simulations, scenario configurations, and result analysis.
"""

import unittest
import numpy as np
from datetime import date, datetime
from decimal import Decimal

from src.portfolio.models import PortfolioSnapshot, Currency, InstrumentType
from src.portfolio.scenarios import (
    PortfolioScenarioEngine, ScenarioConfiguration, ScenarioType,
    MarketAssumptions, AssetClassAssumptions, SimulationResult
)


class TestPortfolioScenarioEngine(unittest.TestCase):
    """Test the portfolio scenario engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = PortfolioScenarioEngine(random_seed=42)  # Fixed seed for reproducibility

        # Create test portfolio snapshot
        self.test_portfolio = PortfolioSnapshot(
            date=date(2024, 1, 1),
            total_value=Decimal("100000.00"),
            cash_balance=Decimal("10000.00"),
            positions_value=Decimal("90000.00"),
            base_currency=Currency.USD,
            positions={},
            cash_balances={Currency.USD: Decimal("10000.00")},
            total_cost_basis=Decimal("95000.00"),
            total_unrealized_pnl=Decimal("5000.00"),
            total_unrealized_pnl_percent=Decimal("5.26")
        )

    def test_create_predefined_scenarios(self):
        """Test creation of predefined scenarios."""
        scenarios = self.engine.create_predefined_scenarios(self.test_portfolio.total_value)

        # Should have all major scenario types
        expected_scenarios = ["optimistic", "likely", "pessimistic", "stress"]
        for scenario_name in expected_scenarios:
            self.assertIn(scenario_name, scenarios)

        # Test optimistic scenario properties
        optimistic = scenarios["optimistic"]
        self.assertEqual(optimistic.scenario_type, ScenarioType.OPTIMISTIC)
        self.assertEqual(optimistic.name, "Bull Market Growth")
        self.assertGreater(optimistic.market_assumptions.expected_return, 0.10)
        self.assertLess(optimistic.market_assumptions.volatility, 0.20)

        # Test stress scenario properties
        stress = scenarios["stress"]
        self.assertEqual(stress.scenario_type, ScenarioType.STRESS)
        self.assertLess(stress.market_assumptions.expected_return, 0.0)
        self.assertGreater(stress.market_assumptions.volatility, 0.40)

        # Test asset class assumptions
        for scenario_config in scenarios.values():
            self.assertIn("STOCK", scenario_config.asset_class_assumptions)
            self.assertIn("BOND", scenario_config.asset_class_assumptions)

            stock_assumptions = scenario_config.asset_class_assumptions["STOCK"]
            self.assertEqual(stock_assumptions.asset_class, InstrumentType.STOCK)
            self.assertGreaterEqual(stock_assumptions.volatility, 0.0)

    def test_scenario_configuration_validation(self):
        """Test scenario configuration validation."""
        # Valid configuration
        valid_config = ScenarioConfiguration(
            scenario_type=ScenarioType.LIKELY,
            name="Test Scenario",
            description="Test description",
            projection_years=3.0,
            market_assumptions=MarketAssumptions(
                expected_return=0.08,
                volatility=0.20
            )
        )

        self.assertEqual(valid_config.projection_years, 3.0)
        self.assertEqual(valid_config.monte_carlo_runs, 1000)  # Default value

        # Test asset class assumptions
        stock_assumptions = AssetClassAssumptions(
            asset_class=InstrumentType.STOCK,
            expected_return=0.10,
            volatility=0.22,
            dividend_yield=0.02
        )

        self.assertEqual(stock_assumptions.asset_class, InstrumentType.STOCK)
        self.assertEqual(stock_assumptions.expected_return, 0.10)
        self.assertEqual(stock_assumptions.dividend_yield, 0.02)

    def test_monte_carlo_simulation_basic(self):
        """Test basic Monte Carlo simulation functionality."""
        # Create a simple scenario
        scenario_config = ScenarioConfiguration(
            scenario_type=ScenarioType.LIKELY,
            name="Test Simulation",
            description="Basic test simulation",
            projection_years=1.0,  # Short simulation for testing
            simulation_steps=52,   # Weekly steps
            monte_carlo_runs=100,  # Fewer runs for speed
            market_assumptions=MarketAssumptions(
                expected_return=0.08,
                volatility=0.20
            )
        )

        # Run simulation
        result = self.engine.run_scenario_simulation(self.test_portfolio, scenario_config)

        # Validate result structure
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.start_value, self.test_portfolio.total_value)
        self.assertEqual(result.scenario_config, scenario_config)

        # Check data dimensions
        self.assertEqual(len(result.dates), 52)  # Weekly steps for 1 year
        self.assertEqual(result.portfolio_values.shape, (100, 52))  # 100 runs, 52 steps
        self.assertEqual(len(result.final_values), 100)

        # Check statistical properties
        self.assertIsInstance(result.mean_trajectory, list)
        self.assertEqual(len(result.mean_trajectory), 52)
        self.assertIsInstance(result.percentiles, dict)
        self.assertIn(0.5, result.percentiles)  # Median should be included

        # Check risk metrics
        self.assertIsInstance(result.probability_of_loss, float)
        self.assertGreaterEqual(result.probability_of_loss, 0.0)
        self.assertLessEqual(result.probability_of_loss, 1.0)

        self.assertIsInstance(result.probability_of_doubling, float)
        self.assertGreaterEqual(result.probability_of_doubling, 0.0)
        self.assertLessEqual(result.probability_of_doubling, 1.0)

        # Check performance metrics
        self.assertEqual(len(result.annualized_returns), 100)
        self.assertEqual(len(result.sharpe_ratios), 100)
        self.assertEqual(len(result.max_drawdowns), 100)

    def test_simulation_result_statistics(self):
        """Test statistical calculations in simulation results."""
        # Create scenario with known parameters
        scenario_config = ScenarioConfiguration(
            scenario_type=ScenarioType.LIKELY,
            name="Statistics Test",
            description="Test statistical calculations",
            projection_years=2.0,
            monte_carlo_runs=500,
            market_assumptions=MarketAssumptions(
                expected_return=0.06,  # 6% expected return
                volatility=0.15       # 15% volatility
            )
        )

        result = self.engine.run_scenario_simulation(self.test_portfolio, scenario_config)

        # Test summary statistics
        summary = result.get_summary_stats()

        required_stats = [
            "mean_final_value", "median_final_value", "std_final_value",
            "min_final_value", "max_final_value", "probability_of_loss",
            "probability_of_doubling", "mean_annualized_return",
            "mean_sharpe_ratio", "mean_max_drawdown"
        ]

        for stat in required_stats:
            self.assertIn(stat, summary)
            self.assertIsInstance(summary[stat], float)

        # Validate statistical relationships
        self.assertLessEqual(summary["min_final_value"], summary["median_final_value"])
        self.assertLessEqual(summary["median_final_value"], summary["max_final_value"])
        self.assertGreaterEqual(summary["std_final_value"], 0.0)

        # With positive expected returns, mean should generally be > start value
        start_value = float(self.test_portfolio.total_value)
        if scenario_config.market_assumptions.expected_return > 0:
            # Allow some tolerance for short simulations with volatility
            self.assertGreater(summary["mean_final_value"], start_value * 0.9)

    def test_scenario_comparison(self):
        """Test comparison of multiple scenarios."""
        scenarios = self.engine.create_predefined_scenarios(self.test_portfolio.total_value)

        # Run simulations for multiple scenarios
        simulation_results = {}
        for name, config in list(scenarios.items())[:2]:  # Test first 2 scenarios
            # Reduce simulation parameters for speed
            config.projection_years = 1.0
            config.monte_carlo_runs = 100
            config.simulation_steps = 52

            simulation_results[name] = self.engine.run_scenario_simulation(
                self.test_portfolio, config
            )

        # Compare scenarios
        comparison = self.engine.compare_scenarios(simulation_results)

        # Validate comparison structure
        self.assertEqual(len(comparison), len(simulation_results))

        for scenario_name, stats in comparison.items():
            self.assertIn(scenario_name, simulation_results)
            self.assertIn("scenario_type", stats)
            self.assertIn("mean_final_value", stats)
            self.assertIn("probability_of_loss", stats)
            self.assertIn("percentile_25", stats)
            self.assertIn("percentile_75", stats)
            self.assertIn("downside_deviation", stats)

        # Test scenario ordering (optimistic should generally outperform pessimistic)
        if "optimistic" in comparison and "pessimistic" in comparison:
            opt_return = comparison["optimistic"]["mean_annualized_return"]
            pess_return = comparison["pessimistic"]["mean_annualized_return"]
            self.assertGreaterEqual(opt_return, pess_return)

    def test_cash_flow_integration(self):
        """Test integration of recurring cash flows."""
        scenario_config = ScenarioConfiguration(
            scenario_type=ScenarioType.LIKELY,
            name="Cash Flow Test",
            description="Test with recurring deposits",
            projection_years=1.0,
            monte_carlo_runs=100,  # Fix validation error
            simulation_steps=12,  # Monthly steps
            recurring_deposits=1000.0,  # $1000 monthly deposits
            market_assumptions=MarketAssumptions(
                expected_return=0.05,
                volatility=0.10
            )
        )

        result = self.engine.run_scenario_simulation(self.test_portfolio, scenario_config)

        # With monthly deposits, final value should be higher than without
        start_value = float(self.test_portfolio.total_value)
        expected_deposits = 12 * 1000  # 12 months * $1000

        # Final values should generally exceed start value + deposits
        # (allowing for some variance due to market returns)
        mean_final = result.get_summary_stats()["mean_final_value"]
        min_expected = start_value + expected_deposits * 0.8  # Conservative estimate

        self.assertGreater(mean_final, min_expected)

    def test_stress_scenario_behavior(self):
        """Test stress scenario specific behavior."""
        stress_config = ScenarioConfiguration(
            scenario_type=ScenarioType.STRESS,
            name="Stress Test",
            description="Test stress scenario mechanics",
            projection_years=2.0,
            monte_carlo_runs=200,
            market_assumptions=MarketAssumptions(
                expected_return=-0.05,  # Negative expected return
                volatility=0.40         # High volatility
            )
        )

        result = self.engine.run_scenario_simulation(self.test_portfolio, stress_config)

        # Stress scenarios should show higher probability of loss
        self.assertGreater(result.probability_of_loss, 0.3)  # At least 30% chance of loss

        # Should have significant drawdowns
        mean_max_drawdown = result.get_summary_stats()["mean_max_drawdown"]
        self.assertLess(mean_max_drawdown, -0.1)  # At least 10% average drawdown

        # Volatility should be reflected in wide distribution
        std_final = result.get_summary_stats()["std_final_value"]
        mean_final = result.get_summary_stats()["mean_final_value"]
        coefficient_of_variation = std_final / mean_final if mean_final > 0 else 0
        self.assertGreater(coefficient_of_variation, 0.2)  # High relative volatility

    def test_percentile_calculations(self):
        """Test percentile calculations in simulation results."""
        scenario_config = ScenarioConfiguration(
            scenario_type=ScenarioType.LIKELY,
            name="Percentile Test",
            description="Test percentile calculations",
            projection_years=1.0,
            monte_carlo_runs=1000,  # Larger sample for better percentile accuracy
            confidence_intervals=[0.05, 0.25, 0.5, 0.75, 0.95],
            market_assumptions=MarketAssumptions(
                expected_return=0.08,
                volatility=0.20
            )
        )

        result = self.engine.run_scenario_simulation(self.test_portfolio, scenario_config)

        # Check percentile structure
        for conf in scenario_config.confidence_intervals:
            self.assertIn(conf, result.percentiles)
            percentile_series = result.percentiles[conf]
            self.assertEqual(len(percentile_series), len(result.dates))

        # Validate percentile ordering (should be monotonic)
        final_step_percentiles = {}
        for conf in scenario_config.confidence_intervals:
            final_step_percentiles[conf] = result.percentiles[conf][-1]

        # Check ordering: 5th percentile < 25th < median < 75th < 95th
        percentile_values = [final_step_percentiles[conf] for conf in sorted(scenario_config.confidence_intervals)]
        for i in range(len(percentile_values) - 1):
            self.assertLessEqual(percentile_values[i], percentile_values[i + 1])

    def test_value_at_risk_calculations(self):
        """Test Value at Risk calculations."""
        scenario_config = ScenarioConfiguration(
            scenario_type=ScenarioType.LIKELY,
            name="VaR Test",
            description="Test VaR calculations",
            projection_years=1.0,
            monte_carlo_runs=1000,
            confidence_intervals=[0.01, 0.05, 0.10],  # Focus on tail risks
            market_assumptions=MarketAssumptions(
                expected_return=0.06,
                volatility=0.25
            )
        )

        result = self.engine.run_scenario_simulation(self.test_portfolio, scenario_config)

        # VaR should be calculated for tail risks only
        for conf in [0.01, 0.05, 0.10]:
            if conf in result.value_at_risk:
                var_value = result.value_at_risk[conf]
                self.assertIsInstance(var_value, float)
                # VaR should be less than the starting value (represents potential loss)
                self.assertLess(var_value, float(self.test_portfolio.total_value))

        # Lower percentiles should have lower VaR values
        if 0.01 in result.value_at_risk and 0.05 in result.value_at_risk:
            self.assertLessEqual(result.value_at_risk[0.01], result.value_at_risk[0.05])

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same random seed."""
        scenario_config = ScenarioConfiguration(
            scenario_type=ScenarioType.LIKELY,
            name="Reproducibility Test",
            description="Test reproducible results",
            projection_years=1.0,
            monte_carlo_runs=100,
            market_assumptions=MarketAssumptions(
                expected_return=0.08,
                volatility=0.20
            )
        )

        # Run same simulation twice with same seed
        engine1 = PortfolioScenarioEngine(random_seed=123)
        engine2 = PortfolioScenarioEngine(random_seed=123)

        result1 = engine1.run_scenario_simulation(self.test_portfolio, scenario_config)
        result2 = engine2.run_scenario_simulation(self.test_portfolio, scenario_config)

        # Results should be identical
        np.testing.assert_array_equal(result1.portfolio_values, result2.portfolio_values)
        self.assertEqual(result1.final_values, result2.final_values)
        self.assertEqual(result1.probability_of_loss, result2.probability_of_loss)


if __name__ == '__main__':
    unittest.main()
