"""
Pytest configuration and shared fixtures for Portfolio Tracker tests.
"""

import sys
import tempfile
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Generator, List

import pytest

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.portfolio.models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    Portfolio,
    Transaction,
    TransactionType,
)
from src.portfolio.storage import FileBasedStorage
from src.portfolio.manager import PortfolioManager
from src.data_providers.manager import DataProviderManager
from src.utils.metrics import FinancialMetricsCalculator


@pytest.fixture
def temp_data_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_instrument() -> FinancialInstrument:
    """Create a sample financial instrument for testing."""
    return FinancialInstrument(
        symbol="AAPL",
        name="Apple Inc.",
        instrument_type=InstrumentType.STOCK,
        currency=Currency.USD,
        exchange="NASDAQ",
        isin="US0378331005",
    )


@pytest.fixture
def sample_crypto_instrument() -> FinancialInstrument:
    """Create a sample cryptocurrency instrument for testing."""
    return FinancialInstrument(
        symbol="BTC-USD",
        name="Bitcoin",
        instrument_type=InstrumentType.CRYPTO,
        currency=Currency.USD,
        exchange="Coinbase",
    )


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Create a sample portfolio for testing."""
    return Portfolio(
        id="test-portfolio-123",
        name="Test Portfolio",
        base_currency=Currency.USD,
        created_at=datetime(2024, 1, 1, 10, 0, 0),
    )


@pytest.fixture
def sample_buy_transaction(sample_instrument: FinancialInstrument) -> Transaction:
    """Create a sample buy transaction for testing."""
    return Transaction(
        id="txn-buy-123",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        instrument=sample_instrument,
        transaction_type=TransactionType.BUY,
        quantity=Decimal("100"),
        price=Decimal("150.00"),
        fees=Decimal("1.00"),
        currency=Currency.USD,
        notes="Sample buy transaction",
    )


@pytest.fixture
def sample_sell_transaction(sample_instrument: FinancialInstrument) -> Transaction:
    """Create a sample sell transaction for testing."""
    return Transaction(
        id="txn-sell-123",
        timestamp=datetime(2024, 1, 20, 14, 15, 0),
        instrument=sample_instrument,
        transaction_type=TransactionType.SELL,
        quantity=Decimal("50"),
        price=Decimal("160.00"),
        fees=Decimal("1.00"),
        currency=Currency.USD,
        notes="Sample sell transaction",
    )


@pytest.fixture
def sample_dividend_transaction(sample_instrument: FinancialInstrument) -> Transaction:
    """Create a sample dividend transaction for testing."""
    return Transaction(
        id="txn-div-123",
        timestamp=datetime(2024, 1, 25, 9, 0, 0),
        instrument=sample_instrument,
        transaction_type=TransactionType.DIVIDEND,
        quantity=Decimal("1"),
        price=Decimal("25.00"),
        fees=Decimal("0"),
        currency=Currency.USD,
        notes="Quarterly dividend",
    )


@pytest.fixture
def portfolio_with_transactions(
    sample_portfolio: Portfolio,
    sample_buy_transaction: Transaction,
    sample_sell_transaction: Transaction,
    sample_dividend_transaction: Transaction,
) -> Portfolio:
    """Create a portfolio with sample transactions."""
    portfolio = sample_portfolio
    portfolio.add_transaction(sample_buy_transaction)
    portfolio.add_transaction(sample_sell_transaction)
    portfolio.add_transaction(sample_dividend_transaction)

    # Add some cash
    portfolio.cash_balances[Currency.USD] = Decimal("1000.00")

    return portfolio


@pytest.fixture
def storage(temp_data_dir: str) -> FileBasedStorage:
    """Create a file-based storage instance for testing."""
    return FileBasedStorage(temp_data_dir)


@pytest.fixture
def mock_data_manager() -> DataProviderManager:
    """Create a mock data manager for testing."""
    # In real tests, we might want to mock the actual API calls
    return DataProviderManager()


@pytest.fixture
def portfolio_manager(storage: FileBasedStorage, mock_data_manager: DataProviderManager) -> PortfolioManager:
    """Create a portfolio manager for testing."""
    return PortfolioManager(storage, mock_data_manager)


@pytest.fixture
def metrics_calculator(mock_data_manager: DataProviderManager) -> FinancialMetricsCalculator:
    """Create a metrics calculator for testing."""
    return FinancialMetricsCalculator(mock_data_manager)


@pytest.fixture
def sample_price_data() -> List[Dict]:
    """Create sample price data for testing."""
    base_date = date(2024, 1, 1)
    prices = []

    for i in range(30):
        price_date = base_date + timedelta(days=i)
        price = 100 + (i * 0.5) + (i % 5)  # Trending up with some volatility

        prices.append({
            "date": price_date,
            "open": price - 0.5,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": 1000000 + (i * 10000),
        })

    return prices


@pytest.fixture
def sample_returns() -> List[float]:
    """Create sample returns data for testing."""
    # Generate 252 trading days of sample returns (roughly 1 year)
    import random
    random.seed(42)  # For reproducible tests

    returns = []
    for _ in range(252):
        # Generate realistic daily returns (mean ~0.08% daily, std ~1.2%)
        daily_return = random.normalvariate(0.0008, 0.012)
        returns.append(daily_return)

    return returns


@pytest.fixture(autouse=True)
def mock_external_apis(monkeypatch):
    """Mock external API calls to avoid network dependencies in tests."""

    def mock_yfinance_price(*args, **kwargs):
        return Decimal("150.00")

    def mock_alpha_vantage_price(*args, **kwargs):
        return Decimal("150.00")

    def mock_exchange_rate(*args, **kwargs):
        return Decimal("1.0")

    # Mock the actual API methods if needed
    # monkeypatch.setattr("yfinance.Ticker.info", lambda x: {"regularMarketPrice": 150.00})
    # This would be expanded based on actual implementation needs


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "external: Tests requiring external services")


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to all tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests that use external APIs
        if "external" in item.name or "api" in item.name.lower():
            item.add_marker(pytest.mark.external)