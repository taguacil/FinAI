"""
Advanced scenario modeling and simulation engine for portfolio projections.
Provides Monte Carlo simulations and scenario-based what-if analysis.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from .models import Currency, PortfolioSnapshot, InstrumentType


class ScenarioType(str, Enum):
    """Types of scenarios for portfolio modeling."""
    OPTIMISTIC = "optimistic"
    LIKELY = "likely"
    PESSIMISTIC = "pessimistic"
    STRESS = "stress"
    CUSTOM = "custom"


class MarketRegime(str, Enum):
    """Market regime classifications."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    RECESSION = "recession"


@dataclass
class MarketAssumptions:
    """Market assumptions for scenario modeling."""

    # Annual return expectations (as decimals, e.g., 0.10 for 10%)
    expected_return: float
    volatility: float  # Annual volatility

    # Correlation assumptions
    equity_correlation: float = 0.7  # Correlation between equity positions
    bond_correlation: float = 0.3    # Correlation between bond positions
    equity_bond_correlation: float = -0.1  # Correlation between equities and bonds

    # Economic indicators
    inflation_rate: float = 0.025  # 2.5% annual inflation
    risk_free_rate: float = 0.02   # 2% risk-free rate

    # Market regime probabilities
    regime_transition_prob: Dict[MarketRegime, float] = None

    def __post_init__(self):
        if self.regime_transition_prob is None:
            self.regime_transition_prob = {
                MarketRegime.BULL_MARKET: 0.4,
                MarketRegime.BEAR_MARKET: 0.15,
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.HIGH_VOLATILITY: 0.1,
                MarketRegime.RECESSION: 0.05
            }


@dataclass
class AssetClassAssumptions:
    """Assumptions for specific asset classes."""

    asset_class: InstrumentType
    expected_return: float
    volatility: float
    correlation_with_market: float = 0.8

    # Asset-specific factors
    dividend_yield: float = 0.0
    expense_ratio: float = 0.0  # For ETFs/funds

    # Risk factors
    max_drawdown_probability: float = 0.05  # 5% chance of severe drawdown
    max_drawdown_magnitude: float = -0.5    # 50% potential drawdown


class ScenarioConfiguration(BaseModel):
    """Configuration for a portfolio scenario."""

    scenario_type: ScenarioType = Field(..., description="Type of scenario")
    name: str = Field(..., description="Human-readable scenario name")
    description: str = Field(..., description="Detailed scenario description")

    # Time horizon
    projection_years: float = Field(default=5.0, ge=0.1, le=50.0, description="Years to project")
    simulation_steps: int = Field(default=252, ge=12, le=2520, description="Steps per year (252 = daily)")

    # Market assumptions
    market_assumptions: MarketAssumptions
    asset_class_assumptions: Dict[str, AssetClassAssumptions] = Field(default_factory=dict)

    # Simulation parameters
    monte_carlo_runs: int = Field(default=1000, ge=100, le=10000, description="Number of MC simulations")
    confidence_intervals: List[float] = Field(default=[0.05, 0.25, 0.5, 0.75, 0.95], description="Confidence levels")

    # Additional cash flows
    recurring_deposits: float = Field(default=0.0, description="Monthly recurring deposits")
    recurring_withdrawals: float = Field(default=0.0, description="Monthly recurring withdrawals")

    # Economic scenarios
    inflation_scenario: float = Field(default=0.025, description="Annual inflation rate")
    currency_shock_probability: float = Field(default=0.02, description="Probability of currency devaluation")


@dataclass
class SimulationResult:
    """Results from a scenario simulation."""

    scenario_config: ScenarioConfiguration
    start_value: Decimal

    # Time series data
    dates: List[date]
    portfolio_values: np.ndarray  # Shape: (monte_carlo_runs, simulation_steps)

    # Statistical summaries
    percentiles: Dict[float, List[float]]  # percentile -> time series
    mean_trajectory: List[float]
    std_trajectory: List[float]

    # Final value statistics
    final_values: List[float]  # Final portfolio values for each simulation
    probability_of_loss: float
    probability_of_doubling: float

    # Risk metrics
    max_drawdowns: List[float]  # Maximum drawdown for each simulation
    value_at_risk: Dict[float, float]  # VaR at different confidence levels

    # Performance metrics
    annualized_returns: List[float]
    sharpe_ratios: List[float]

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the scenario."""
        return {
            "mean_final_value": float(np.mean(self.final_values)),
            "median_final_value": float(np.median(self.final_values)),
            "std_final_value": float(np.std(self.final_values)),
            "min_final_value": float(np.min(self.final_values)),
            "max_final_value": float(np.max(self.final_values)),
            "probability_of_loss": self.probability_of_loss,
            "probability_of_doubling": self.probability_of_doubling,
            "mean_annualized_return": float(np.mean(self.annualized_returns)),
            "mean_sharpe_ratio": float(np.mean(self.sharpe_ratios)),
            "mean_max_drawdown": float(np.mean(self.max_drawdowns)),
            "var_95": self.value_at_risk.get(0.95, 0.0),
            "var_99": self.value_at_risk.get(0.99, 0.0)
        }


class PortfolioScenarioEngine:
    """Advanced portfolio scenario modeling and simulation engine."""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the scenario engine.

        Args:
            random_seed: Seed for reproducible results
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.logger = logging.getLogger(__name__)

    def create_predefined_scenarios(self, base_portfolio_value: Decimal) -> Dict[str, ScenarioConfiguration]:
        """Create predefined scenario configurations.

        Args:
            base_portfolio_value: Current portfolio value for scaling

        Returns:
            Dictionary mapping scenario names to configurations
        """
        scenarios = {}

        # Optimistic Scenario (Bull Market)
        scenarios["optimistic"] = ScenarioConfiguration(
            scenario_type=ScenarioType.OPTIMISTIC,
            name="Bull Market Growth",
            description="Strong economic growth, low volatility, favorable market conditions",
            projection_years=5.0,
            market_assumptions=MarketAssumptions(
                expected_return=0.12,  # 12% annual return
                volatility=0.15,       # 15% volatility
                equity_correlation=0.6,
                bond_correlation=0.2,
                equity_bond_correlation=-0.1,
                inflation_rate=0.02,
                risk_free_rate=0.025
            ),
            asset_class_assumptions={
                "STOCK": AssetClassAssumptions(
                    asset_class=InstrumentType.STOCK,
                    expected_return=0.14,
                    volatility=0.18,
                    dividend_yield=0.02
                ),
                "ETF": AssetClassAssumptions(
                    asset_class=InstrumentType.ETF,
                    expected_return=0.11,
                    volatility=0.16,
                    expense_ratio=0.005
                ),
                "BOND": AssetClassAssumptions(
                    asset_class=InstrumentType.BOND,
                    expected_return=0.06,
                    volatility=0.05,
                    dividend_yield=0.04
                )
            },
            monte_carlo_runs=1000
        )

        # Likely Scenario (Historical Average)
        scenarios["likely"] = ScenarioConfiguration(
            scenario_type=ScenarioType.LIKELY,
            name="Historical Average",
            description="Markets perform in line with long-term historical averages",
            projection_years=5.0,
            market_assumptions=MarketAssumptions(
                expected_return=0.08,  # 8% annual return
                volatility=0.20,       # 20% volatility
                equity_correlation=0.7,
                bond_correlation=0.3,
                equity_bond_correlation=-0.1,
                inflation_rate=0.025,
                risk_free_rate=0.02
            ),
            asset_class_assumptions={
                "STOCK": AssetClassAssumptions(
                    asset_class=InstrumentType.STOCK,
                    expected_return=0.10,
                    volatility=0.22,
                    dividend_yield=0.02
                ),
                "ETF": AssetClassAssumptions(
                    asset_class=InstrumentType.ETF,
                    expected_return=0.08,
                    volatility=0.18,
                    expense_ratio=0.007
                ),
                "BOND": AssetClassAssumptions(
                    asset_class=InstrumentType.BOND,
                    expected_return=0.04,
                    volatility=0.06,
                    dividend_yield=0.035
                )
            },
            monte_carlo_runs=1000
        )

        # Pessimistic Scenario (Bear Market)
        scenarios["pessimistic"] = ScenarioConfiguration(
            scenario_type=ScenarioType.PESSIMISTIC,
            name="Economic Downturn",
            description="Economic recession, market stress, high volatility",
            projection_years=5.0,
            market_assumptions=MarketAssumptions(
                expected_return=0.02,  # 2% annual return
                volatility=0.30,       # 30% volatility
                equity_correlation=0.85,  # Higher correlation in stress
                bond_correlation=0.4,
                equity_bond_correlation=0.1,  # Correlations break down
                inflation_rate=0.035,
                risk_free_rate=0.015
            ),
            asset_class_assumptions={
                "STOCK": AssetClassAssumptions(
                    asset_class=InstrumentType.STOCK,
                    expected_return=0.03,
                    volatility=0.35,
                    dividend_yield=0.025,
                    max_drawdown_probability=0.15,
                    max_drawdown_magnitude=-0.4
                ),
                "ETF": AssetClassAssumptions(
                    asset_class=InstrumentType.ETF,
                    expected_return=0.02,
                    volatility=0.28,
                    expense_ratio=0.008
                ),
                "BOND": AssetClassAssumptions(
                    asset_class=InstrumentType.BOND,
                    expected_return=0.025,
                    volatility=0.08,
                    dividend_yield=0.03
                )
            },
            monte_carlo_runs=1000
        )

        # Stress Scenario (Market Crash)
        scenarios["stress"] = ScenarioConfiguration(
            scenario_type=ScenarioType.STRESS,
            name="Market Crash",
            description="Severe market crash followed by slow recovery",
            projection_years=5.0,
            market_assumptions=MarketAssumptions(
                expected_return=-0.05,  # -5% annual return
                volatility=0.45,        # 45% volatility
                equity_correlation=0.95,  # Very high correlation in crash
                bond_correlation=0.6,
                equity_bond_correlation=0.3,
                inflation_rate=0.01,  # Deflationary
                risk_free_rate=0.005
            ),
            asset_class_assumptions={
                "STOCK": AssetClassAssumptions(
                    asset_class=InstrumentType.STOCK,
                    expected_return=-0.02,
                    volatility=0.50,
                    dividend_yield=0.015,
                    max_drawdown_probability=0.8,
                    max_drawdown_magnitude=-0.6
                ),
                "ETF": AssetClassAssumptions(
                    asset_class=InstrumentType.ETF,
                    expected_return=-0.03,
                    volatility=0.42,
                    expense_ratio=0.01
                ),
                "BOND": AssetClassAssumptions(
                    asset_class=InstrumentType.BOND,
                    expected_return=0.015,
                    volatility=0.12,
                    dividend_yield=0.02
                )
            },
            monte_carlo_runs=1000
        )

        return scenarios

    def run_scenario_simulation(
        self,
        current_portfolio: PortfolioSnapshot,
        scenario_config: ScenarioConfiguration
    ) -> SimulationResult:
        """Run a Monte Carlo simulation for a given scenario.

        Args:
            current_portfolio: Current portfolio state
            scenario_config: Scenario configuration

        Returns:
            Simulation results
        """
        self.logger.info(f"Running scenario simulation: {scenario_config.name}")

        # Set up simulation parameters
        start_value = current_portfolio.total_value
        years = scenario_config.projection_years
        steps_per_year = scenario_config.simulation_steps
        total_steps = int(years * steps_per_year)
        mc_runs = scenario_config.monte_carlo_runs
        dt = 1.0 / steps_per_year  # Time step in years

        # Generate dates
        start_date = current_portfolio.date
        dates = [start_date + timedelta(days=int(i * 365.25 / steps_per_year)) for i in range(total_steps)]

        # Initialize portfolio value matrix
        portfolio_values = np.zeros((mc_runs, total_steps))
        portfolio_values[:, 0] = float(start_value)

        # Get market assumptions
        market_assumptions = scenario_config.market_assumptions
        mu = market_assumptions.expected_return
        sigma = market_assumptions.volatility

        # Set random seed for this simulation if specified
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Run Monte Carlo simulations
        for run in range(mc_runs):
            if run % 100 == 0:
                self.logger.debug(f"Running simulation {run}/{mc_runs}")

            # Generate correlated random shocks
            random_shocks = self._generate_correlated_returns(
                total_steps, market_assumptions, dt
            )

            # Simulate portfolio evolution
            for step in range(1, total_steps):
                # Apply return shock
                return_shock = random_shocks[step]

                # Add recurring cash flows
                cash_flow = self._calculate_cash_flow(step, dt, scenario_config)

                # Update portfolio value
                current_value = portfolio_values[run, step - 1]
                growth = current_value * return_shock
                new_value = current_value + growth + cash_flow

                # Apply any stress events
                if scenario_config.scenario_type == ScenarioType.STRESS:
                    new_value = self._apply_stress_events(new_value, step, total_steps, dt)

                portfolio_values[run, step] = max(new_value, 0)  # Prevent negative values

        # Calculate statistics
        percentiles = {}
        for conf in scenario_config.confidence_intervals:
            percentiles[conf] = np.percentile(portfolio_values, conf * 100, axis=0).tolist()

        mean_trajectory = np.mean(portfolio_values, axis=0).tolist()
        std_trajectory = np.std(portfolio_values, axis=0).tolist()

        # Final value statistics
        final_values = portfolio_values[:, -1].tolist()
        start_val_float = float(start_value)
        probability_of_loss = np.mean(np.array(final_values) < start_val_float)
        probability_of_doubling = np.mean(np.array(final_values) >= 2 * start_val_float)

        # Calculate risk metrics
        max_drawdowns = self._calculate_max_drawdowns(portfolio_values)
        value_at_risk = self._calculate_var(final_values, scenario_config.confidence_intervals)

        # Calculate performance metrics
        annualized_returns = self._calculate_annualized_returns(portfolio_values, years)
        sharpe_ratios = self._calculate_sharpe_ratios(
            annualized_returns, market_assumptions.risk_free_rate, sigma
        )

        return SimulationResult(
            scenario_config=scenario_config,
            start_value=start_value,
            dates=dates,
            portfolio_values=portfolio_values,
            percentiles=percentiles,
            mean_trajectory=mean_trajectory,
            std_trajectory=std_trajectory,
            final_values=final_values,
            probability_of_loss=probability_of_loss,
            probability_of_doubling=probability_of_doubling,
            max_drawdowns=max_drawdowns,
            value_at_risk=value_at_risk,
            annualized_returns=annualized_returns,
            sharpe_ratios=sharpe_ratios
        )

    def _generate_correlated_returns(
        self,
        total_steps: int,
        market_assumptions: MarketAssumptions,
        dt: float
    ) -> np.ndarray:
        """Generate correlated returns using geometric Brownian motion."""
        mu = market_assumptions.expected_return
        sigma = market_assumptions.volatility

        # Generate random normal shocks
        random_normals = np.random.normal(0, 1, total_steps)

        # Apply geometric Brownian motion formula
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * random_normals

        # Calculate returns
        log_returns = drift + diffusion
        returns = np.exp(log_returns) - 1

        return returns

    def _calculate_cash_flow(self, step: int, dt: float, config: ScenarioConfiguration) -> float:
        """Calculate recurring cash flows for a given time step."""
        # Monthly cash flows (assuming steps are daily)
        if step % int(1.0 / (dt * 12)) == 0:  # Monthly
            return config.recurring_deposits - config.recurring_withdrawals
        return 0.0

    def _apply_stress_events(self, value: float, step: int, total_steps: int, dt: float) -> float:
        """Apply stress events for stress scenarios."""
        # Apply early crash (first year)
        if step < int(1.0 / dt):  # First year
            crash_probability = 0.05 * dt  # 5% annual probability
            if np.random.random() < crash_probability:
                crash_magnitude = np.random.uniform(-0.3, -0.1)  # 10-30% crash
                value *= (1 + crash_magnitude)

        return value

    def _calculate_max_drawdowns(self, portfolio_values: np.ndarray) -> List[float]:
        """Calculate maximum drawdown for each simulation run."""
        max_drawdowns = []

        for run in range(portfolio_values.shape[0]):
            values = portfolio_values[run, :]
            running_max = np.maximum.accumulate(values)
            drawdown = (values - running_max) / running_max
            max_drawdown = np.min(drawdown)
            max_drawdowns.append(max_drawdown)

        return max_drawdowns

    def _calculate_var(self, final_values: List[float], confidence_intervals: List[float]) -> Dict[float, float]:
        """Calculate Value at Risk at different confidence levels."""
        var = {}
        sorted_values = np.sort(final_values)

        for conf in confidence_intervals:
            if conf < 0.5:  # Lower tail
                percentile_idx = int(conf * len(sorted_values))
                var[conf] = sorted_values[percentile_idx]

        return var

    def _calculate_annualized_returns(self, portfolio_values: np.ndarray, years: float) -> List[float]:
        """Calculate annualized returns for each simulation run."""
        annualized_returns = []

        for run in range(portfolio_values.shape[0]):
            start_value = portfolio_values[run, 0]
            end_value = portfolio_values[run, -1]

            if start_value > 0:
                total_return = (end_value / start_value) - 1
                annualized_return = (1 + total_return) ** (1 / years) - 1
                annualized_returns.append(annualized_return)
            else:
                annualized_returns.append(0.0)

        return annualized_returns

    def _calculate_sharpe_ratios(
        self,
        annualized_returns: List[float],
        risk_free_rate: float,
        volatility: float
    ) -> List[float]:
        """Calculate Sharpe ratios for each simulation run."""
        sharpe_ratios = []

        for annual_return in annualized_returns:
            if volatility > 0:
                sharpe_ratio = (annual_return - risk_free_rate) / volatility
                sharpe_ratios.append(sharpe_ratio)
            else:
                sharpe_ratios.append(0.0)

        return sharpe_ratios

    def compare_scenarios(
        self,
        simulation_results: Dict[str, SimulationResult]
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple scenarios and provide summary comparison.

        Args:
            simulation_results: Dictionary mapping scenario names to results

        Returns:
            Comparison metrics for each scenario
        """
        comparison = {}

        for scenario_name, result in simulation_results.items():
            stats = result.get_summary_stats()

            # Add scenario-specific insights
            stats["scenario_type"] = result.scenario_config.scenario_type.value
            stats["projection_years"] = result.scenario_config.projection_years

            # Calculate additional comparative metrics
            if result.final_values:
                stats["percentile_25"] = float(np.percentile(result.final_values, 25))
                stats["percentile_75"] = float(np.percentile(result.final_values, 75))
                stats["downside_deviation"] = self._calculate_downside_deviation(result.final_values, float(result.start_value))

            comparison[scenario_name] = stats

        return comparison

    def _calculate_downside_deviation(self, final_values: List[float], target: float) -> float:
        """Calculate downside deviation relative to a target value."""
        below_target = [v for v in final_values if v < target]
        if not below_target:
            return 0.0

        squared_deviations = [(target - v) ** 2 for v in below_target]
        return np.sqrt(np.mean(squared_deviations))
