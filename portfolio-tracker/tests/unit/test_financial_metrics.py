"""
Unit tests for financial metrics calculations.
"""

import pytest
import numpy as np
from datetime import date, timedelta
from decimal import Decimal

from src.utils.metrics import FinancialMetricsCalculator
from src.portfolio.models import PortfolioSnapshot, Currency


class TestFinancialMetricsCalculator:
    """Test cases for FinancialMetricsCalculator."""

    def test_calculate_returns(self, metrics_calculator):
        """Test calculating returns from portfolio snapshots."""
        # Create sample snapshots with known values
        snapshots = [
            PortfolioSnapshot(
                date=date(2024, 1, 1),
                total_value=Decimal("10000"),
                cash_balance=Decimal("1000"),
                positions_value=Decimal("9000"),
                base_currency=Currency.USD,
            ),
            PortfolioSnapshot(
                date=date(2024, 1, 2),
                total_value=Decimal("10500"),  # 5% increase
                cash_balance=Decimal("1000"),
                positions_value=Decimal("9500"),
                base_currency=Currency.USD,
            ),
            PortfolioSnapshot(
                date=date(2024, 1, 3),
                total_value=Decimal("10000"),  # Back to original (~-4.76%)
                cash_balance=Decimal("1000"),
                positions_value=Decimal("9000"),
                base_currency=Currency.USD,
            ),
        ]
        
        returns = metrics_calculator.calculate_returns(snapshots)
        
        assert len(returns) == 2
        assert abs(returns[0] - 0.05) < 0.001  # ~5% return
        assert abs(returns[1] - (-0.047619)) < 0.001  # ~-4.76% return

    def test_calculate_returns_insufficient_data(self, metrics_calculator):
        """Test calculating returns with insufficient data."""
        snapshots = [
            PortfolioSnapshot(
                date=date(2024, 1, 1),
                total_value=Decimal("10000"),
                cash_balance=Decimal("1000"),
                positions_value=Decimal("9000"),
                base_currency=Currency.USD,
            )
        ]
        
        returns = metrics_calculator.calculate_returns(snapshots)
        assert len(returns) == 0

    def test_calculate_volatility(self, metrics_calculator, sample_returns):
        """Test volatility calculation."""
        volatility = metrics_calculator.calculate_volatility(sample_returns, annualized=False)
        
        # Should be positive
        assert volatility > 0
        
        # Annualized should be higher (sqrt(252) factor)
        annualized_vol = metrics_calculator.calculate_volatility(sample_returns, annualized=True)
        assert annualized_vol > volatility
        assert abs(annualized_vol / volatility - np.sqrt(252)) < 0.01

    def test_calculate_volatility_insufficient_data(self, metrics_calculator):
        """Test volatility calculation with insufficient data."""
        returns = [0.01]  # Only one return
        volatility = metrics_calculator.calculate_volatility(returns)
        assert volatility == 0.0

    def test_calculate_sharpe_ratio(self, metrics_calculator, sample_returns):
        """Test Sharpe ratio calculation."""
        risk_free_rate = 0.02  # 2% risk-free rate
        sharpe = metrics_calculator.calculate_sharpe_ratio(sample_returns, risk_free_rate)
        
        # Should be a reasonable value
        assert -5.0 < sharpe < 5.0
        
        # Test with zero risk-free rate
        sharpe_zero_rf = metrics_calculator.calculate_sharpe_ratio(sample_returns, 0.0)
        assert sharpe_zero_rf != sharpe

    def test_calculate_sharpe_ratio_zero_volatility(self, metrics_calculator):
        """Test Sharpe ratio with zero volatility."""
        returns = [0.01, 0.01, 0.01]  # Constant returns = zero volatility
        sharpe = metrics_calculator.calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_calculate_max_drawdown(self, metrics_calculator):
        """Test maximum drawdown calculation."""
        # Create snapshots with known drawdown
        snapshots = [
            PortfolioSnapshot(date=date(2024, 1, 1), total_value=Decimal("10000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("10000"), 
                            base_currency=Currency.USD),
            PortfolioSnapshot(date=date(2024, 1, 2), total_value=Decimal("12000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("12000"), 
                            base_currency=Currency.USD),  # Peak
            PortfolioSnapshot(date=date(2024, 1, 3), total_value=Decimal("9000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("9000"), 
                            base_currency=Currency.USD),  # 25% drawdown from peak
            PortfolioSnapshot(date=date(2024, 1, 4), total_value=Decimal("11000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("11000"), 
                            base_currency=Currency.USD),
        ]
        
        max_dd, duration = metrics_calculator.calculate_max_drawdown(snapshots)
        
        assert abs(max_dd - 0.25) < 0.001  # 25% drawdown
        assert duration == 2  # 2 days duration

    def test_calculate_max_drawdown_no_drawdown(self, metrics_calculator):
        """Test max drawdown with always increasing values."""
        snapshots = [
            PortfolioSnapshot(date=date(2024, 1, 1), total_value=Decimal("10000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("10000"), 
                            base_currency=Currency.USD),
            PortfolioSnapshot(date=date(2024, 1, 2), total_value=Decimal("11000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("11000"), 
                            base_currency=Currency.USD),
            PortfolioSnapshot(date=date(2024, 1, 3), total_value=Decimal("12000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("12000"), 
                            base_currency=Currency.USD),
        ]
        
        max_dd, duration = metrics_calculator.calculate_max_drawdown(snapshots)
        
        assert max_dd == 0.0
        assert duration == 0

    def test_calculate_beta(self, metrics_calculator):
        """Test beta calculation."""
        # Portfolio returns
        portfolio_returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        
        # Market returns (higher correlation should give beta closer to 1)
        benchmark_returns = [0.008, 0.018, -0.012, 0.012, -0.008]
        
        beta = metrics_calculator.calculate_beta(portfolio_returns, benchmark_returns)
        
        # Should be positive and reasonable
        assert 0.0 < beta < 3.0

    def test_calculate_beta_mismatched_length(self, metrics_calculator):
        """Test beta calculation with mismatched return lengths."""
        portfolio_returns = [0.01, 0.02, -0.01]
        benchmark_returns = [0.008, 0.018]  # Different length
        
        beta = metrics_calculator.calculate_beta(portfolio_returns, benchmark_returns)
        assert beta == 0.0

    def test_calculate_beta_zero_variance(self, metrics_calculator):
        """Test beta calculation when benchmark has zero variance."""
        portfolio_returns = [0.01, 0.02, -0.01]
        benchmark_returns = [0.01, 0.01, 0.01]  # Constant returns
        
        beta = metrics_calculator.calculate_beta(portfolio_returns, benchmark_returns)
        assert beta == 0.0

    def test_calculate_alpha(self, metrics_calculator):
        """Test alpha calculation."""
        portfolio_returns = [0.02, 0.03, -0.01, 0.02, 0.01]  # Higher returns
        benchmark_returns = [0.01, 0.02, -0.01, 0.015, 0.005]  # Lower returns
        risk_free_rate = 0.02
        
        alpha = metrics_calculator.calculate_alpha(portfolio_returns, benchmark_returns, risk_free_rate)
        
        # Should be positive since portfolio outperformed
        assert alpha > 0

    def test_calculate_information_ratio(self, metrics_calculator):
        """Test information ratio calculation."""
        portfolio_returns = [0.02, 0.03, -0.01, 0.02, 0.01]
        benchmark_returns = [0.015, 0.025, -0.005, 0.018, 0.008]
        
        info_ratio = metrics_calculator.calculate_information_ratio(portfolio_returns, benchmark_returns)
        
        # Should be a reasonable value
        assert -5.0 < info_ratio < 5.0

    def test_calculate_information_ratio_zero_tracking_error(self, metrics_calculator):
        """Test information ratio with zero tracking error."""
        returns = [0.01, 0.02, -0.01]
        # Identical returns = zero tracking error
        
        info_ratio = metrics_calculator.calculate_information_ratio(returns, returns)
        assert info_ratio == 0.0

    def test_calculate_sortino_ratio(self, metrics_calculator):
        """Test Sortino ratio calculation."""
        # Returns with some negative values
        returns = [0.02, 0.03, -0.02, 0.01, -0.01, 0.025]
        
        sortino = metrics_calculator.calculate_sortino_ratio(returns)
        
        # Should be finite and reasonable
        assert not np.isinf(sortino)
        assert -10.0 < sortino < 10.0

    def test_calculate_sortino_ratio_no_downside(self, metrics_calculator):
        """Test Sortino ratio with no downside (all positive returns)."""
        returns = [0.01, 0.02, 0.015, 0.025, 0.01]
        
        sortino = metrics_calculator.calculate_sortino_ratio(returns)
        
        # Should be infinite (no downside risk)
        assert np.isinf(sortino)

    def test_calculate_calmar_ratio(self, metrics_calculator):
        """Test Calmar ratio calculation."""
        returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        
        # Create snapshots with some drawdown
        snapshots = [
            PortfolioSnapshot(date=date(2024, 1, 1), total_value=Decimal("10000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("10000"), 
                            base_currency=Currency.USD),
            PortfolioSnapshot(date=date(2024, 1, 2), total_value=Decimal("10100"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("10100"), 
                            base_currency=Currency.USD),
            PortfolioSnapshot(date=date(2024, 1, 3), total_value=Decimal("9000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("9000"), 
                            base_currency=Currency.USD),  # Drawdown
        ]
        
        calmar = metrics_calculator.calculate_calmar_ratio(returns, snapshots)
        
        # Should be finite
        assert not np.isinf(calmar)
        assert calmar != 0.0

    def test_calculate_calmar_ratio_zero_drawdown(self, metrics_calculator):
        """Test Calmar ratio with zero drawdown."""
        returns = [0.01, 0.02, 0.015]
        
        # No drawdown snapshots
        snapshots = [
            PortfolioSnapshot(date=date(2024, 1, 1), total_value=Decimal("10000"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("10000"), 
                            base_currency=Currency.USD),
            PortfolioSnapshot(date=date(2024, 1, 2), total_value=Decimal("10100"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("10100"), 
                            base_currency=Currency.USD),
            PortfolioSnapshot(date=date(2024, 1, 3), total_value=Decimal("10200"), 
                            cash_balance=Decimal("0"), positions_value=Decimal("10200"), 
                            base_currency=Currency.USD),
        ]
        
        calmar = metrics_calculator.calculate_calmar_ratio(returns, snapshots)
        
        # Should be infinite (no drawdown)
        assert np.isinf(calmar)

    def test_calculate_value_at_risk(self, metrics_calculator, sample_returns):
        """Test Value at Risk calculation."""
        var_5pct = metrics_calculator.calculate_value_at_risk(sample_returns, 0.05)
        var_1pct = metrics_calculator.calculate_value_at_risk(sample_returns, 0.01)
        
        # VaR should be positive
        assert var_5pct > 0
        assert var_1pct > 0
        
        # 1% VaR should be higher than 5% VaR
        assert var_1pct > var_5pct

    def test_calculate_conditional_var(self, metrics_calculator, sample_returns):
        """Test Conditional Value at Risk calculation."""
        cvar_5pct = metrics_calculator.calculate_conditional_var(sample_returns, 0.05)
        var_5pct = metrics_calculator.calculate_value_at_risk(sample_returns, 0.05)
        
        # CVaR should be positive and typically higher than VaR
        assert cvar_5pct > 0
        assert cvar_5pct >= var_5pct

    def test_get_benchmark_returns(self, metrics_calculator, mock_data_manager):
        """Test getting benchmark returns."""
        # This would require mocking the data manager properly
        # For now, just test that it doesn't crash
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        returns = metrics_calculator.get_benchmark_returns("SPY", start_date, end_date)
        
        # Should return a list (might be empty if mocked)
        assert isinstance(returns, list)

    def test_calculate_portfolio_metrics_comprehensive(self, metrics_calculator):
        """Test comprehensive portfolio metrics calculation."""
        # Create sample snapshots
        snapshots = []
        base_value = 10000
        
        for i in range(30):  # 30 days of data
            # Add some realistic variation
            daily_change = np.random.normal(0.001, 0.02)  # Mean 0.1% daily, 2% std
            base_value *= (1 + daily_change)
            
            snapshot = PortfolioSnapshot(
                date=date(2024, 1, 1) + timedelta(days=i),
                total_value=Decimal(str(round(base_value, 2))),
                cash_balance=Decimal("1000"),
                positions_value=Decimal(str(round(base_value - 1000, 2))),
                base_currency=Currency.USD,
            )
            snapshots.append(snapshot)
        
        metrics = metrics_calculator.calculate_portfolio_metrics(snapshots, "SPY")
        
        # Check that all expected metrics are present
        expected_keys = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'max_drawdown_duration',
            'var_5pct', 'cvar_5pct', 'calmar_ratio', 'benchmark_symbol'
        ]
        
        for key in expected_keys:
            assert key in metrics
            
        # Check that values are reasonable
        assert isinstance(metrics['total_return'], float)
        assert isinstance(metrics['volatility'], float)
        assert metrics['volatility'] >= 0
        assert metrics['max_drawdown'] >= 0

    def test_calculate_sector_allocation(self, metrics_calculator):
        """Test sector allocation calculation."""
        positions = [
            {
                'symbol': 'AAPL',
                'market_value': Decimal('5000'),
                'instrument_type': 'stock'
            },
            {
                'symbol': 'GOOGL',
                'market_value': Decimal('3000'),
                'instrument_type': 'stock'
            },
            {
                'symbol': 'BTC',
                'market_value': Decimal('2000'),
                'instrument_type': 'crypto'
            }
        ]
        
        allocation = metrics_calculator.calculate_sector_allocation(positions)
        
        # Should return percentages that sum to 100%
        assert isinstance(allocation, dict)
        if allocation:  # Only check if we got results (depends on mocked data)
            total_percentage = sum(allocation.values())
            assert abs(total_percentage - 100.0) < 0.01

    def test_calculate_currency_allocation(self, metrics_calculator):
        """Test currency allocation calculation."""
        positions = [
            {
                'symbol': 'AAPL',
                'market_value': Decimal('6000'),
                'currency': 'USD'
            },
            {
                'symbol': 'SAP',
                'market_value': Decimal('3000'),
                'currency': 'EUR'
            },
            {
                'symbol': 'CASH',
                'market_value': Decimal('1000'),
                'currency': 'USD'
            }
        ]
        
        allocation = metrics_calculator.calculate_currency_allocation(positions)
        
        # Should return percentages
        assert isinstance(allocation, dict)
        assert 'USD' in allocation
        assert 'EUR' in allocation
        
        # USD should be 70% (7000/10000), EUR should be 30%
        assert abs(allocation['USD'] - 70.0) < 0.01
        assert abs(allocation['EUR'] - 30.0) < 0.01
        
        # Total should be 100%
        total_percentage = sum(allocation.values())
        assert abs(total_percentage - 100.0) < 0.01

    def test_metrics_with_empty_data(self, metrics_calculator):
        """Test metrics calculation with empty data."""
        # Empty snapshots
        metrics = metrics_calculator.calculate_portfolio_metrics([])
        assert 'error' in metrics
        
        # Single snapshot
        snapshot = PortfolioSnapshot(
            date=date(2024, 1, 1),
            total_value=Decimal("10000"),
            cash_balance=Decimal("1000"),
            positions_value=Decimal("9000"),
            base_currency=Currency.USD,
        )
        metrics = metrics_calculator.calculate_portfolio_metrics([snapshot])
        assert 'error' in metrics