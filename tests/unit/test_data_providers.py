"""
Unit tests for data providers.
"""

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from src.data_providers.alpha_vantage import AlphaVantageProvider
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
    def mock_alpha_provider(self):
        """Create a mock Alpha Vantage provider."""
        provider = Mock()
        provider.name = "Alpha Vantage"
        provider.supports_instrument_type.return_value = True
        return provider

    @pytest.fixture
    def manager(self, mock_yahoo_provider, mock_alpha_provider):
        """Create a data provider manager with mocked providers."""
        with patch(
            "src.data_providers.manager.YahooFinanceProvider"
        ) as mock_yahoo_class:
            with patch(
                "src.data_providers.manager.AlphaVantageProvider"
            ) as mock_alpha_class:
                mock_yahoo_class.return_value = mock_yahoo_provider
                mock_alpha_class.return_value = mock_alpha_provider

                manager = DataProviderManager()
                # Manually set providers for testing
                manager.providers = [mock_yahoo_provider, mock_alpha_provider]
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

    def test_get_current_price_fallback(
        self, manager, mock_yahoo_provider, mock_alpha_provider
    ):
        """Test fallback to second provider when first fails."""
        mock_yahoo_provider.get_current_price.side_effect = Exception("Yahoo error")
        mock_alpha_provider.get_current_price.return_value = Decimal("150.00")

        price = manager.get_current_price("AAPL")

        assert price == Decimal("150.00")
        mock_yahoo_provider.get_current_price.assert_called_once_with("AAPL")
        mock_alpha_provider.get_current_price.assert_called_once_with("AAPL")

    def test_get_current_price_all_providers_fail(
        self, manager, mock_yahoo_provider, mock_alpha_provider
    ):
        """Test when all providers fail."""
        mock_yahoo_provider.get_current_price.side_effect = Exception("Yahoo error")
        mock_alpha_provider.get_current_price.side_effect = Exception("Alpha error")

        price = manager.get_current_price("AAPL")

        assert price is None

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

    def test_search_instruments_combine_results(
        self, manager, mock_yahoo_provider, mock_alpha_provider
    ):
        """Test combining search results from multiple providers."""
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
        alpha_results = [
            InstrumentInfo(
                symbol="MSFT",
                name="Microsoft Corp",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
            ),
            InstrumentInfo(
                symbol="AAPL",
                name="Apple Inc.",
                instrument_type=InstrumentType.STOCK,
                currency=Currency.USD,
            ),  # Duplicate
        ]

        mock_yahoo_provider.search_instruments.return_value = yahoo_results
        mock_alpha_provider.search_instruments.return_value = alpha_results

        results = manager.search_instruments("tech")

        # Should have 3 unique results (AAPL, TSLA, MSFT)
        assert len(results) == 3
        symbols = [r.symbol for r in results]
        assert "AAPL" in symbols
        assert "TSLA" in symbols
        assert "MSFT" in symbols

    def test_get_exchange_rate_caching(self, manager, mock_yahoo_provider):
        """Test exchange rate retrieval with caching."""
        # Mock Alpha Vantage provider to return the rate (it's tried first for forex)
        mock_alpha_provider = Mock()
        mock_alpha_provider.name = "Alpha Vantage"
        mock_alpha_provider.get_exchange_rate.return_value = Decimal("0.85")

        # Replace the providers list to include our mock
        manager.providers = [mock_alpha_provider, mock_yahoo_provider]

        # First call
        rate1 = manager.get_exchange_rate(Currency.USD, Currency.EUR)
        assert rate1 == Decimal("0.85")

        # Second call should use cache
        rate2 = manager.get_exchange_rate(Currency.USD, Currency.EUR)
        assert rate2 == Decimal("0.85")

        # Should only be called once due to caching
        assert mock_alpha_provider.get_exchange_rate.call_count == 1

    def test_get_exchange_rate_same_currency(self, manager):
        """Test exchange rate for same currency."""
        rate = manager.get_exchange_rate(Currency.USD, Currency.USD)
        assert rate == Decimal("1")

    def test_get_provider_status(
        self, manager, mock_yahoo_provider, mock_alpha_provider
    ):
        """Test getting provider status."""
        mock_yahoo_provider.get_current_price.return_value = Decimal("150.00")
        mock_alpha_provider.get_current_price.return_value = None

        status = manager.get_provider_status()

        assert status["Yahoo Finance"] is True
        assert status["Alpha Vantage"] is False

    def test_clear_cache(self, manager, mock_yahoo_provider):
        """Test clearing cache."""
        # Add some data to cache
        mock_info = InstrumentInfo(
            symbol="AAPL",
            name="Apple Inc.",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )
        manager._instrument_cache["AAPL"] = mock_info
        manager._exchange_rate_cache["USD_EUR"] = Decimal("0.85")

        # Clear cache
        manager.clear_cache()

        assert len(manager._instrument_cache) == 0
        assert len(manager._exchange_rate_cache) == 0

    def test_get_providers_for_instrument_type(
        self, manager, mock_yahoo_provider, mock_alpha_provider
    ):
        """Test getting providers for specific instrument type."""
        mock_yahoo_provider.supports_instrument_type.return_value = True
        mock_alpha_provider.supports_instrument_type.return_value = False

        providers = manager.get_providers_for_instrument(InstrumentType.STOCK)

        assert len(providers) == 1
        assert providers[0].name == "Yahoo Finance"


class TestYahooFinanceProvider:
    """Test the YahooFinanceProvider."""

    @pytest.fixture
    def provider(self):
        """Create a Yahoo Finance provider instance."""
        return YahooFinanceProvider()

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_success(self, mock_ticker_class, provider):
        """Test successful current price retrieval."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "currentPrice": 150.00,
            "regularMarketPrice": 149.50,
            "previousClose": 148.00,
        }
        mock_ticker_class.return_value = mock_ticker

        price = provider.get_current_price("AAPL")

        assert price == Decimal("150.00")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_fallback_to_regular_market(
        self, mock_ticker_class, provider
    ):
        """Test fallback to regular market price when current price is not available."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "currentPrice": None,
            "regularMarketPrice": 149.50,
            "previousClose": 148.00,
        }
        mock_ticker_class.return_value = mock_ticker

        price = provider.get_current_price("AAPL")

        assert price == Decimal("149.50")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_fallback_to_previous_close(
        self, mock_ticker_class, provider
    ):
        """Test fallback to previous close when other prices are not available."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "currentPrice": None,
            "regularMarketPrice": 148.00,  # This must be available for the logic to work
            "previousClose": 148.00,
        }
        mock_ticker_class.return_value = mock_ticker

        price = provider.get_current_price("AAPL")

        assert price == Decimal("148.00")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_no_price_available(self, mock_ticker_class, provider):
        """Test handling when no price is available."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "currentPrice": None,
            "regularMarketPrice": None,  # This will cause InvalidSymbolError
            "previousClose": None,
        }
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(
            InvalidSymbolError, match="Invalid or delisted symbol: AAPL"
        ):
            provider.get_current_price("AAPL")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_invalid_symbol(self, mock_ticker_class, provider):
        """Test handling invalid symbol."""
        mock_ticker = Mock()
        mock_ticker.info = None
        mock_ticker_class.return_value = mock_ticker

        with pytest.raises(InvalidSymbolError):
            provider.get_current_price("INVALID")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_rate_limit_error(self, mock_ticker_class, provider):
        """Test handling rate limit errors."""
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_ticker_class.side_effect = Exception("rate limit exceeded")

        with pytest.raises(RateLimitError):
            provider.get_current_price("AAPL")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_connection_error(self, mock_ticker_class, provider):
        """Test handling connection errors."""
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_ticker_class.side_effect = Exception("connection failed")

        with pytest.raises(ConnectionError):
            provider.get_current_price("AAPL")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_current_price_timeout_error(self, mock_ticker_class, provider):
        """Test handling timeout errors."""
        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_ticker_class.side_effect = Exception("timeout")

        with pytest.raises(TimeoutError):
            provider.get_current_price("AAPL")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_instrument_info_stock(self, mock_ticker_class, provider):
        """Test getting instrument info for a stock."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "quoteType": "EQUITY",
            "currency": "USD",
            "exchange": "NASDAQ",
            "isin": "US0378331005",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 2500000000000,
            "longBusinessSummary": "Apple Inc. designs, manufactures, and markets smartphones...",
        }
        mock_ticker_class.return_value = mock_ticker

        info = provider.get_instrument_info("AAPL")

        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.instrument_type == InstrumentType.STOCK
        assert info.currency == Currency.USD
        assert info.exchange == "NASDAQ"
        assert info.isin == "US0378331005"
        assert info.sector == "Technology"
        assert info.industry == "Consumer Electronics"
        assert info.market_cap == Decimal("2500000000000")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_instrument_info_etf(self, mock_ticker_class, provider):
        """Test getting instrument info for an ETF."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "symbol": "SPY",
            "longName": "SPDR S&P 500 ETF Trust",
            "quoteType": "ETF",
            "currency": "USD",
            "exchange": "PCX",
        }
        mock_ticker_class.return_value = mock_ticker

        info = provider.get_instrument_info("SPY")

        assert info.instrument_type == InstrumentType.ETF

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_instrument_info_crypto(self, mock_ticker_class, provider):
        """Test getting instrument info for cryptocurrency."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "symbol": "BTC-USD",
            "longName": "Bitcoin USD",
            "quoteType": "CRYPTOCURRENCY",
            "currency": "USD",
        }
        mock_ticker_class.return_value = mock_ticker

        info = provider.get_instrument_info("BTC-USD")

        assert info.instrument_type == InstrumentType.CRYPTO

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_instrument_info_fallback_name(self, mock_ticker_class, provider):
        """Test fallback to short name when long name is not available."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "quoteType": "EQUITY",
            "currency": "USD",
        }
        mock_ticker_class.return_value = mock_ticker

        info = provider.get_instrument_info("AAPL")

        assert info.name == "Apple Inc."

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_instrument_info_symbol_fallback(self, mock_ticker_class, provider):
        """Test fallback to symbol when no name is available."""
        mock_ticker = Mock()
        mock_ticker.info = {"symbol": "AAPL", "quoteType": "EQUITY", "currency": "USD"}
        mock_ticker_class.return_value = mock_ticker

        info = provider.get_instrument_info("AAPL")

        assert info.name == "AAPL"

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_instrument_info_currency_fallback(self, mock_ticker_class, provider):
        """Test fallback to USD when currency is not available."""
        mock_ticker = Mock()
        mock_ticker.info = {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "quoteType": "EQUITY",
        }
        mock_ticker_class.return_value = mock_ticker

        info = provider.get_instrument_info("AAPL")

        assert info.currency == Currency.USD

    def test_supports_instrument_type(self, provider):
        """Test instrument type support."""
        assert provider.supports_instrument_type(InstrumentType.STOCK) is True
        assert provider.supports_instrument_type(InstrumentType.ETF) is True
        assert provider.supports_instrument_type(InstrumentType.MUTUAL_FUND) is True
        assert provider.supports_instrument_type(InstrumentType.CRYPTO) is True
        assert provider.supports_instrument_type(InstrumentType.BOND) is False
        assert provider.supports_instrument_type(InstrumentType.OPTION) is False

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_exchange_rate_usd_to_other(self, mock_ticker_class, provider):
        """Test getting exchange rate from USD to other currency."""
        mock_ticker = Mock()
        mock_ticker.info = {"regularMarketPrice": 0.85}
        mock_ticker_class.return_value = mock_ticker

        rate = provider.get_exchange_rate(Currency.USD, Currency.EUR)

        # Should return 1/0.85 = 1.176...
        assert rate == Decimal("1") / Decimal("0.85")

    @patch("src.data_providers.yahoo_finance.yf.Ticker")
    def test_get_exchange_rate_other_to_usd(self, mock_ticker_class, provider):
        """Test getting exchange rate from other currency to USD."""
        mock_ticker = Mock()
        mock_ticker.info = {"regularMarketPrice": 1.18}
        mock_ticker_class.return_value = mock_ticker

        rate = provider.get_exchange_rate(Currency.EUR, Currency.USD)

        assert rate == Decimal("1.18")

    def test_get_exchange_rate_same_currency(self, provider):
        """Test exchange rate for same currency."""
        rate = provider.get_exchange_rate(Currency.USD, Currency.USD)
        assert rate == Decimal("1")

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


class TestAlphaVantageProvider:
    """Test the AlphaVantageProvider."""

    @pytest.fixture
    def provider(self):
        """Create an Alpha Vantage provider instance."""
        return AlphaVantageProvider("test_api_key")

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_current_price_success(self, mock_get, provider):
        """Test successful current price retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {"Global Quote": {"05. price": "150.00"}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        price = provider.get_current_price("AAPL")

        assert price == Decimal("150.00")

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_current_price_no_price(self, mock_get, provider):
        """Test handling when no price is available."""
        mock_response = Mock()
        mock_response.json.return_value = {"Global Quote": {}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        price = provider.get_current_price("AAPL")

        assert price is None

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_current_price_api_error(self, mock_get, provider):
        """Test handling API errors."""
        mock_response = Mock()
        mock_response.json.return_value = {"Error Message": "Invalid API call"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with pytest.raises(DataProviderError):
            provider.get_current_price("AAPL")

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_current_price_rate_limit(self, mock_get, provider):
        """Test handling rate limit errors."""
        mock_response = Mock()
        mock_response.json.return_value = {"Note": "API call frequency limit exceeded"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with pytest.raises(RateLimitError):
            provider.get_current_price("AAPL")

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_current_price_request_exception(self, mock_get, provider):
        """Test handling request exceptions."""
        mock_get.side_effect = requests.RequestException("Network error")

        price = provider.get_current_price("AAPL")

        assert price is None

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_historical_prices_success(self, mock_get, provider):
        """Test successful historical price retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "150.00",
                    "2. high": "152.00",
                    "3. low": "149.00",
                    "4. close": "151.00",
                    "5. volume": "1000000",
                },
                "2024-01-16": {
                    "1. open": "151.00",
                    "2. high": "153.00",
                    "3. low": "150.00",
                    "4. close": "152.00",
                    "5. volume": "1100000",
                },
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        prices = provider.get_historical_prices(
            "AAPL", date(2024, 1, 15), date(2024, 1, 16)
        )

        assert len(prices) == 2
        assert prices[0].date == date(2024, 1, 15)
        assert prices[0].open_price == Decimal("150.00")
        assert prices[0].close_price == Decimal("151.00")
        assert prices[1].date == date(2024, 1, 16)
        assert prices[1].close_price == Decimal("152.00")

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_instrument_info_with_overview(self, mock_get, provider):
        """Test getting instrument info with overview data."""
        # First call for quote
        mock_response1 = Mock()
        mock_response1.json.return_value = {"Global Quote": {"01. symbol": "AAPL"}}
        mock_response1.raise_for_status.return_value = None

        # Second call for overview
        mock_response2 = Mock()
        mock_response2.json.return_value = {
            "Symbol": "AAPL",
            "Name": "Apple Inc.",
            "Exchange": "NASDAQ",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "MarketCapitalization": "2500000000000",
            "Description": "Apple Inc. designs, manufactures...",
        }
        mock_response2.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response1, mock_response2]

        info = provider.get_instrument_info("AAPL")

        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.exchange == "NASDAQ"
        assert info.sector == "Technology"
        assert info.industry == "Consumer Electronics"
        assert info.market_cap == Decimal("2500000000000")

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_instrument_info_fallback(self, mock_get, provider):
        """Test getting instrument info with fallback to basic data."""
        mock_response = Mock()
        mock_response.json.return_value = {"Global Quote": {"01. symbol": "AAPL"}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        info = provider.get_instrument_info("AAPL")

        assert info.symbol == "AAPL"
        assert info.name == "AAPL"  # Fallback to symbol
        assert info.instrument_type == InstrumentType.STOCK
        assert info.currency == Currency.USD

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_search_instruments_success(self, mock_get, provider):
        """Test successful instrument search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "bestMatches": [
                {
                    "1. symbol": "AAPL",
                    "2. name": "Apple Inc.",
                    "4. region": "United States",
                },
                {
                    "1. symbol": "MSFT",
                    "2. name": "Microsoft Corporation",
                    "4. region": "United States",
                },
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        results = provider.search_instruments("Apple")

        assert len(results) == 2
        assert results[0].symbol == "AAPL"
        assert results[0].name == "Apple Inc."
        assert results[0].currency == Currency.USD
        assert results[1].symbol == "MSFT"
        assert results[1].name == "Microsoft Corporation"

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_search_instruments_region_mapping(self, mock_get, provider):
        """Test region to currency mapping."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "bestMatches": [
                {
                    "1. symbol": "RY",
                    "2. name": "Royal Bank of Canada",
                    "4. region": "Canada",
                },
                {
                    "1. symbol": "VOD",
                    "2. name": "Vodafone Group",
                    "4. region": "United Kingdom",
                },
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        results = provider.search_instruments("bank")

        assert results[0].currency == Currency.CAD  # Canada
        assert results[1].currency == Currency.GBP  # United Kingdom

    @patch("src.data_providers.alpha_vantage.requests.get")
    def test_get_exchange_rate_success(self, mock_get, provider):
        """Test successful exchange rate retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Realtime Currency Exchange Rate": {"5. Exchange Rate": "0.85"}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        rate = provider.get_exchange_rate(Currency.USD, Currency.EUR)

        assert rate == Decimal("0.85")

    def test_get_exchange_rate_same_currency(self, provider):
        """Test exchange rate for same currency."""
        rate = provider.get_exchange_rate(Currency.USD, Currency.USD)
        assert rate == Decimal("1")

    def test_supports_instrument_type(self, provider):
        """Test instrument type support."""
        assert provider.supports_instrument_type(InstrumentType.STOCK) is True
        assert provider.supports_instrument_type(InstrumentType.ETF) is True
        assert provider.supports_instrument_type(InstrumentType.MUTUAL_FUND) is False
        assert provider.supports_instrument_type(InstrumentType.CRYPTO) is False

    def test_rate_limiting(self, provider):
        """Test rate limiting functionality."""
        import time

        start_time = time.time()

        # First call
        provider._rate_limit()
        # Second call immediately after should be delayed
        provider._rate_limit()

        elapsed = time.time() - start_time

        # Should have been delayed by at least 12 seconds (Alpha Vantage limit)
        assert elapsed >= 12
