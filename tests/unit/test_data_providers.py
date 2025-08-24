"""
Unit tests for data providers.
"""

import time
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from src.data_providers.base import (
    ConnectionError,
    DataProviderError,
    InstrumentInfo,
    InvalidSymbolError,
    PriceData,
    RateLimitError,
    TimeoutError,
)
from src.data_providers.manager import DataProviderManager
from src.data_providers.yahoo_finance import YahooFinanceProvider
from src.portfolio.models import Currency, InstrumentType


class TestDataProviderManager:
    """Test the DataProviderManager."""

    @pytest.fixture
    def mock_yahoo_provider(self):
        """Create a mock Yahoo Finance provider."""
        provider = Mock()
        provider.name = "Yahoo Finance"
        provider.supports_instrument_type.return_value = True
        return provider

    @pytest.fixture
    def manager(self, mock_yahoo_provider):
        """Create a data provider manager with mocked providers."""
        with patch(
            "src.data_providers.manager.YahooFinanceProvider"
        ) as mock_yahoo_class:
            mock_yahoo_class.return_value = mock_yahoo_provider

            manager = DataProviderManager()
            # Manually set providers for testing
            manager.providers = [mock_yahoo_provider]
            return manager

    def test_normalize_symbol(self, manager):
        """Test symbol normalization."""
        assert manager._normalize_symbol("AAPL") == "AAPL"
        assert manager._normalize_symbol("$AAPL") == "AAPL"
        assert manager._normalize_symbol("  aapl  ") == "AAPL"
        assert manager._normalize_symbol("$TSLA ") == "TSLA"

    def test_get_current_price_success(self, manager, mock_yahoo_provider):
        """Test successful current price retrieval."""
        mock_yahoo_provider.get_current_price.return_value = Decimal("150.00")

        price = manager.get_current_price("AAPL")

        assert price == Decimal("150.00")
        mock_yahoo_provider.get_current_price.assert_called_once_with("AAPL")

    def test_get_current_price_fallback(self, manager, mock_yahoo_provider):
        """Test fallback when provider fails."""
        mock_yahoo_provider.get_current_price.side_effect = Exception("Yahoo error")

        price = manager.get_current_price("AAPL")

        assert price is None
        mock_yahoo_provider.get_current_price.assert_called_once_with("AAPL")

    def test_get_current_price_provider_fails(self, manager, mock_yahoo_provider):
        """Test when provider fails."""
        mock_yahoo_provider.get_current_price.side_effect = Exception("Yahoo error")

        price = manager.get_current_price("AAPL")

        assert price is None
        mock_yahoo_provider.get_current_price.assert_called_once_with("AAPL")

    def test_get_instrument_info_with_caching(self, manager, mock_yahoo_provider):
        """Test instrument info retrieval with caching."""
        mock_info = InstrumentInfo(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )
        mock_yahoo_provider.get_instrument_info.return_value = mock_info

        # First call should hit the provider
        info1 = manager.get_instrument_info("AAPL")
        assert info1 == mock_info

        # Second call should use cache
        info2 = manager.get_instrument_info("AAPL")
        assert info2 == mock_info

        # Should only be called once due to caching
        assert mock_yahoo_provider.get_instrument_info.call_count == 1

    def test_get_instrument_info_force_refresh(self, manager, mock_yahoo_provider):
        """Test instrument info retrieval with force refresh."""
        mock_info = InstrumentInfo(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )
        mock_yahoo_provider.get_instrument_info.return_value = mock_info

        # First call
        manager.get_instrument_info("AAPL")
        # Second call with force refresh
        manager.get_instrument_info("AAPL", force_refresh=True)

        # Should be called twice due to force refresh
        assert mock_yahoo_provider.get_instrument_info.call_count == 2

    def test_search_instruments_single_provider(self, manager, mock_yahoo_provider):
        """Test search results from single provider."""
        yahoo_results = [
            InstrumentInfo(
                symbol="AAPL",
                name="Apple Inc.",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
            ),
            InstrumentInfo(
                symbol="TSLA",
                name="Tesla Inc.",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
            ),
        ]

        mock_yahoo_provider.search_instruments.return_value = yahoo_results

        results = manager.search_instruments("tech")

        # Should have 2 results (AAPL, TSLA)
        assert len(results) == 2
        symbols = [r.symbol for r in results]
        assert "AAPL" in symbols
        assert "TSLA" in symbols

    def test_get_exchange_rate_caching(self, manager, mock_yahoo_provider):
        """Test exchange rate caching."""
        mock_yahoo_provider.get_exchange_rate.return_value = Decimal("0.85")

        # First call should hit the provider
        rate1 = manager.get_exchange_rate(Currency.USD, Currency.EUR)
        assert rate1 == Decimal("0.85")

        # Second call should use cache
        rate2 = manager.get_exchange_rate(Currency.USD, Currency.EUR)
        assert rate2 == Decimal("0.85")

        # Should only be called once due to caching
        assert mock_yahoo_provider.get_exchange_rate.call_count == 1

    def test_get_exchange_rate_force_refresh(self, manager, mock_yahoo_provider):
        """Test exchange rate retrieval with force refresh."""
        mock_yahoo_provider.get_exchange_rate.return_value = Decimal("0.85")

        # First call
        manager.get_exchange_rate(Currency.USD, Currency.EUR)
        # Second call with force refresh
        manager.get_exchange_rate(Currency.USD, Currency.EUR, force_refresh=True)

        # Should be called twice due to force refresh
        assert mock_yahoo_provider.get_exchange_rate.call_count == 2


class TestYahooFinanceProvider:
    """Test the YahooFinanceProvider."""

    @pytest.fixture
    def provider(self):
        """Create a Yahoo Finance provider instance."""
        return YahooFinanceProvider()

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_success(self, mock_ticker, provider):
        """Test successful current price retrieval."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"regularMarketPrice": 150.00}
        mock_ticker.return_value = mock_ticker_instance

        price = provider.get_current_price("AAPL")

        assert price == Decimal("150.00")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_no_price(self, mock_ticker, provider):
        """Test handling when no price is available."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"regularMarketPrice": None}
        mock_ticker.return_value = mock_ticker_instance

        with pytest.raises(InvalidSymbolError):
            provider.get_current_price("AAPL")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_instrument_info_success(self, mock_ticker, provider):
        """Test successful instrument info retrieval."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "quoteType": "EQUITY",
            "currency": "USD",
            "exchange": "NMS",
        }
        mock_ticker.return_value = mock_ticker_instance

        info = provider.get_instrument_info("AAPL")

        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.instrument_type == InstrumentType.STOCK
        assert info.currency == Currency.USD

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_search_instruments_success(self, mock_ticker, provider):
        """Test successful instrument search."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            "symbol": "APPLE",
            "longName": "Apple Inc.",
            "quoteType": "EQUITY",
            "currency": "USD",
        }
        mock_ticker.return_value = mock_ticker_instance

        results = provider.search_instruments("Apple")

        assert len(results) == 1
        assert results[0].symbol == "APPLE"
        assert results[0].name == "Apple Inc."
        assert results[0].currency == Currency.USD

    def test_supports_instrument_type(self, provider):
        """Test instrument type support."""
        assert provider.supports_instrument_type(InstrumentType.STOCK) is True
        assert provider.supports_instrument_type(InstrumentType.ETF) is True
        assert provider.supports_instrument_type(InstrumentType.MUTUAL_FUND) is True
        assert provider.supports_instrument_type(InstrumentType.CRYPTO) is True
        assert provider.supports_instrument_type(InstrumentType.BOND) is True

    def test_rate_limiting(self, provider):
        """Test rate limiting functionality."""
        import time

        start_time = time.time()

        # First call
        provider._rate_limit()
        # Second call immediately after should be delayed
        provider._rate_limit()

        elapsed = time.time() - start_time

        # Should have been delayed by at least 0.1 seconds
        assert elapsed >= 0.1
