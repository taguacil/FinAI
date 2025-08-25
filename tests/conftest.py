"""Test configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

from src.portfolio.storage import FileBasedStorage
from src.data_providers.manager import DataProviderManager


@pytest.fixture(scope="function")
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def mock_storage(temp_data_dir):
    """Create a mock storage with temporary directory."""
    return FileBasedStorage(data_dir=temp_data_dir)


@pytest.fixture(scope="function")
def mock_data_manager():
    """Create a mock data manager."""
    return Mock(spec=DataProviderManager)


@pytest.fixture(scope="function")
def mock_portfolio_manager(mock_storage, mock_data_manager):
    """Create a portfolio manager with mocked dependencies."""
    from src.portfolio.manager import PortfolioManager
    return PortfolioManager(storage=mock_storage, data_manager=mock_data_manager)


@pytest.fixture(scope="function")
def sample_instruments():
    """Create sample financial instruments for testing."""
    from src.portfolio.models import FinancialInstrument, InstrumentType, Currency

    return {
        "AAPL": FinancialInstrument(
            symbol="AAPL",
            name="Apple Inc.",
            isin="US0378331005",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD
        ),
        "MSFT": FinancialInstrument(
            symbol="MSFT",
            name="Microsoft Corporation",
            isin="US5949181045",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD
        ),
        "SPY": FinancialInstrument(
            symbol="SPY",
            name="SPDR S&P 500 ETF Trust",
            isin="US78462F1030",
            instrument_type=InstrumentType.ETF,
            currency=Currency.USD
        ),
        "TLT": FinancialInstrument(
            symbol="TLT",
            name="iShares 20+ Year Treasury Bond ETF",
            isin="US4642876555",
            instrument_type=InstrumentType.BOND,
            currency=Currency.USD
        ),
        "EURSTOCK": FinancialInstrument(
            symbol="EURSTOCK",
            name="European Stock Fund",
            isin="DE0001234567",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.EUR
        )
    }


@pytest.fixture(scope="function")
def sample_transactions(sample_instruments):
    """Create sample transactions for testing."""
    from src.portfolio.models import Transaction, TransactionType, Currency
    from datetime import datetime, timedelta
    from decimal import Decimal

    base_time = datetime.now() - timedelta(days=10)

    return [
        Transaction(
            instrument=sample_instruments["AAPL"],
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("150.00"),
            timestamp=base_time + timedelta(days=1),
            currency=Currency.USD,
            notes="Initial purchase"
        ),
        Transaction(
            instrument=sample_instruments["MSFT"],
            transaction_type=TransactionType.BUY,
            quantity=Decimal("50"),
            price=Decimal("300.00"),
            timestamp=base_time + timedelta(days=2),
            currency=Currency.USD,
            notes="Microsoft purchase"
        ),
        Transaction(
            instrument=sample_instruments["SPY"],
            transaction_type=TransactionType.BUY,
            quantity=Decimal("200"),
            price=Decimal("450.00"),
            timestamp=base_time + timedelta(days=3),
            currency=Currency.USD,
            notes="ETF purchase"
        ),
        Transaction(
            instrument=sample_instruments["AAPL"],
            transaction_type=TransactionType.SELL,
            quantity=Decimal("25"),
            price=Decimal("160.00"),
            timestamp=base_time + timedelta(days=5),
            currency=Currency.USD,
            notes="Partial sale"
        ),
        Transaction(
            instrument=sample_instruments["EURSTOCK"],
            transaction_type=TransactionType.BUY,
            quantity=Decimal("100"),
            price=Decimal("100.00"),
            timestamp=base_time + timedelta(days=4),
            currency=Currency.EUR,
            notes="European stock"
        )
    ]


@pytest.fixture(scope="function")
def sample_portfolio(sample_instruments, sample_transactions):
    """Create a sample portfolio with transactions for testing."""
    from src.portfolio.models import Portfolio, Currency
    from datetime import datetime, timedelta

    portfolio = Portfolio(
        id="test-portfolio-123",
        name="Test Portfolio",
        base_currency=Currency.USD,
        created_at=datetime.now() - timedelta(days=15)
    )

    # Add transactions
    portfolio.transactions = sample_transactions

    # Add some cash balances
    portfolio.cash_balances = {
        Currency.USD: Decimal("5000.00"),
        Currency.EUR: Decimal("1000.00")
    }

    return portfolio


@pytest.fixture(scope="function")
def mock_exchange_rates():
    """Create mock exchange rates for testing."""
    from src.portfolio.models import Currency
    from decimal import Decimal

    return {
        (Currency.EUR, Currency.USD): Decimal("1.2"),
        (Currency.GBP, Currency.USD): Decimal("1.3"),
        (Currency.CHF, Currency.USD): Decimal("1.1"),
        (Currency.JPY, Currency.USD): Decimal("0.0067"),
        (Currency.CAD, Currency.USD): Decimal("0.75"),
        (Currency.AUD, Currency.USD): Decimal("0.65")
    }


@pytest.fixture(scope="function")
def mock_historical_prices():
    """Create mock historical prices for testing."""
    from decimal import Decimal
    from datetime import date, timedelta

    base_date = date.today() - timedelta(days=30)
    prices = {}

    # Generate 30 days of prices
    for i in range(30):
        current_date = base_date + timedelta(days=i)
        prices[current_date] = {
            "AAPL": Decimal("150.00") + Decimal(str(i * 0.5)),  # Gradual increase
            "MSFT": Decimal("300.00") + Decimal(str(i * 1.0)),  # Gradual increase
            "SPY": Decimal("450.00") + Decimal(str(i * 0.3)),   # Gradual increase
            "TLT": Decimal("90.00") + Decimal(str(i * 0.1)),    # Gradual increase
            "EURSTOCK": Decimal("100.00") + Decimal(str(i * 0.2))  # Gradual increase
        }

    return prices


@pytest.fixture(scope="function")
def mock_data_provider():
    """Create a mock data provider for testing."""
    from src.data_providers.base import BaseDataProvider
    from src.portfolio.models import Currency
    from decimal import Decimal

    mock_provider = Mock(spec=BaseDataProvider)
    mock_provider.name = "Mock Provider"

    # Mock exchange rate method
    def mock_get_exchange_rate(from_currency, to_currency):
        if from_currency == to_currency:
            return Decimal("1.0")
        elif from_currency == Currency.EUR and to_currency == Currency.USD:
            return Decimal("1.2")
        elif from_currency == Currency.GBP and to_currency == Currency.USD:
            return Decimal("1.3")
        return None

    mock_provider.get_exchange_rate.side_effect = mock_get_exchange_rate

    # Mock historical exchange rate method
    def mock_get_historical_fx_rate_on(date, from_currency, to_currency):
        if from_currency == to_currency:
            return Decimal("1.0")
        elif from_currency == Currency.EUR and to_currency == Currency.USD:
            return Decimal("1.2")
        elif from_currency == Currency.GBP and to_currency == Currency.USD:
            return Decimal("1.3")
        return None

    mock_provider.get_historical_fx_rate_on.side_effect = mock_get_historical_fx_rate_on

    return mock_provider


@pytest.fixture(scope="function")
def caplog(caplog):
    """Enhanced caplog fixture with better formatting."""
    caplog.set_level("DEBUG")
    return caplog


# Add markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "transaction_fallbacks: marks tests for transaction-based fallback functionality"
    )
    config.addinivalue_line(
        "markers", "position_fallbacks: marks tests for Position model fallback functionality"
    )
    config.addinivalue_line(
        "markers", "portfolio_fallbacks: marks tests for Portfolio model fallback functionality"
    )
    config.addinivalue_line(
        "markers", "snapshot_creation: marks tests for snapshot creation functionality"
    )
    config.addinivalue_line(
        "markers", "exchange_rates: marks tests for exchange rate functionality"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests that require integration testing"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests that are pure unit tests"
    )
