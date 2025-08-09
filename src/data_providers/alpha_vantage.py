"""
Alpha Vantage data provider implementation for stocks and forex.
"""

import os
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

import requests

from ..portfolio.models import Currency, InstrumentType
from .base import (
    BaseDataProvider,
    DataProviderError,
    InstrumentInfo,
    PriceData,
    RateLimitError,
)


class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage data provider for stocks and forex data."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Alpha Vantage provider."""
        self.api_key = api_key or os.getenv(
            "ALPHA_VANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY_PLACEHOLDER"
        )
        self.base_url = "https://www.alphavantage.co/query"
        self.name = "Alpha Vantage"
        self.last_request_time = 0
        self.min_request_interval = 12  # Alpha Vantage free tier: 5 calls per minute

    def _rate_limit(self):
        """Rate limiting for Alpha Vantage API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make API request to Alpha Vantage."""
        self._rate_limit()

        params["apikey"] = self.api_key

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                raise DataProviderError(
                    f"Alpha Vantage API error: {data['Error Message']}"
                )

            if "Note" in data and "API call frequency" in data["Note"]:
                raise RateLimitError("Alpha Vantage rate limit exceeded")

            return data

        except requests.RequestException as e:
            print(f"Error making Alpha Vantage API request: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol."""
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol}

        data = self._make_request(params)
        if not data:
            return None

        try:
            quote = data.get("Global Quote", {})
            price = quote.get("05. price")

            if price:
                return Decimal(str(price))
            return None

        except (KeyError, ValueError) as e:
            print(f"Error parsing current price for {symbol}: {e}")
            return None

    def get_historical_prices(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[PriceData]:
        """Get historical price data for a symbol."""
        # Determine if we need daily or intraday data
        days_diff = (end_date - start_date).days

        if days_diff <= 100:
            function = "TIME_SERIES_DAILY"
            time_series_key = "Time Series (Daily)"
        else:
            function = "TIME_SERIES_DAILY"
            time_series_key = "Time Series (Daily)"

        params = {"function": function, "symbol": symbol, "outputsize": "full"}

        data = self._make_request(params)
        if not data:
            return []

        try:
            time_series = data.get(time_series_key, {})
            price_data = []

            for date_str, price_info in time_series.items():
                try:
                    price_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                    # Filter by date range
                    if price_date < start_date or price_date > end_date:
                        continue

                    price_data.append(
                        PriceData(
                            symbol=symbol,
                            date=price_date,
                            open_price=Decimal(str(price_info["1. open"])),
                            high_price=Decimal(str(price_info["2. high"])),
                            low_price=Decimal(str(price_info["3. low"])),
                            close_price=Decimal(str(price_info["4. close"])),
                            volume=int(price_info["5. volume"]),
                        )
                    )

                except (KeyError, ValueError, TypeError) as e:
                    print(
                        f"Error processing price data for {symbol} on {date_str}: {e}"
                    )
                    continue

            return sorted(price_data, key=lambda x: x.date)

        except (KeyError, ValueError) as e:
            print(f"Error parsing historical prices for {symbol}: {e}")
            return []

    def get_instrument_info(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get basic information about an instrument."""
        # Alpha Vantage doesn't provide comprehensive company info in free tier
        # We'll get basic info from a quote and make educated guesses
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol}

        data = self._make_request(params)
        if not data:
            return None

        try:
            quote = data.get("Global Quote", {})
            if not quote:
                return None

            # Basic info from quote
            symbol_clean = quote.get("01. symbol", symbol).upper()

            # Try to get overview data (premium feature, might not work with free API key)
            overview_params = {"function": "OVERVIEW", "symbol": symbol}
            overview_data = self._make_request(overview_params)

            if overview_data and overview_data.get("Symbol"):
                overview = overview_data
                return InstrumentInfo(
                    symbol=symbol_clean,
                    name=overview.get("Name", symbol_clean),
                    instrument_type=InstrumentType.STOCK,  # Alpha Vantage mainly covers stocks
                    currency=Currency.USD,  # Most Alpha Vantage data is USD
                    exchange=overview.get("Exchange"),
                    sector=overview.get("Sector"),
                    industry=overview.get("Industry"),
                    market_cap=(
                        Decimal(str(overview["MarketCapitalization"]))
                        if overview.get("MarketCapitalization")
                        and overview["MarketCapitalization"] != "None"
                        else None
                    ),
                    description=overview.get("Description"),
                )
            else:
                # Fallback to basic info
                return InstrumentInfo(
                    symbol=symbol_clean,
                    name=symbol_clean,
                    instrument_type=InstrumentType.STOCK,
                    currency=Currency.USD,
                )

        except (KeyError, ValueError) as e:
            print(f"Error getting instrument info for {symbol}: {e}")
            return None

    def search_instruments(self, query: str) -> List[InstrumentInfo]:
        """Search for instruments by keywords."""
        params = {"function": "SYMBOL_SEARCH", "keywords": query}

        data = self._make_request(params)
        if not data:
            return []

        try:
            matches = data.get("bestMatches", [])
            results = []

            for match in matches[:10]:  # Limit to top 10 results
                try:
                    # Map region to currency (rough approximation)
                    region = match.get("4. region", "United States")
                    if "United States" in region:
                        currency = Currency.USD
                    elif "Canada" in region:
                        currency = Currency.CAD
                    elif "United Kingdom" in region:
                        currency = Currency.GBP
                    elif "Europe" in region or "Germany" in region:
                        currency = Currency.EUR
                    else:
                        currency = Currency.USD

                    results.append(
                        InstrumentInfo(
                            symbol=match["1. symbol"],
                            name=match["2. name"],
                            instrument_type=InstrumentType.STOCK,
                            currency=currency,
                            exchange=match.get("4. region"),
                        )
                    )
                except (KeyError, ValueError) as e:
                    print(f"Error processing search result: {e}")
                    continue

            return results

        except (KeyError, ValueError) as e:
            print(f"Error parsing search results: {e}")
            return []

    def get_exchange_rate(
        self, from_currency: Currency, to_currency: Currency
    ) -> Optional[Decimal]:
        """Get current exchange rate between two currencies."""
        if from_currency == to_currency:
            return Decimal("1")

        params = {
            "function": "FX_RATE",
            "from_symbol": from_currency.value,
            "to_symbol": to_currency.value,
        }

        data = self._make_request(params)
        if not data:
            return None

        try:
            fx_data = data.get("Realtime Currency Exchange Rate", {})
            rate = fx_data.get("5. Exchange Rate")

            if rate:
                return Decimal(str(rate))
            return None

        except (KeyError, ValueError) as e:
            print(f"Error getting exchange rate {from_currency} to {to_currency}: {e}")
            return None

    def supports_instrument_type(self, instrument_type: InstrumentType) -> bool:
        """Check if provider supports a specific instrument type."""
        supported_types = {
            InstrumentType.STOCK,
            InstrumentType.ETF,  # Limited ETF support
        }
        return instrument_type in supported_types
