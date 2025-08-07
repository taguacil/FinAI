"""
Data provider manager for coordinating multiple financial data sources.
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

from ..portfolio.models import Currency, InstrumentType
from .alpha_vantage import AlphaVantageProvider
from .base import BaseDataProvider, InstrumentInfo, PriceData
from .yahoo_finance import YahooFinanceProvider


class DataProviderManager:
    """Manages multiple data providers and routes requests appropriately."""

    def __init__(self, alpha_vantage_api_key: Optional[str] = None):
        """Initialize the data provider manager."""
        self.providers: List[BaseDataProvider] = []
        self.provider_priorities: Dict[InstrumentType, List[str]] = {}

        # Initialize providers
        self._setup_providers(alpha_vantage_api_key)
        self._setup_priorities()

        # Cache for instrument info to avoid repeated API calls
        self._instrument_cache: Dict[str, InstrumentInfo] = {}
        self._exchange_rate_cache: Dict[str, Decimal] = {}

    def _setup_providers(self, alpha_vantage_api_key: Optional[str]):
        """Set up available data providers."""
        # Yahoo Finance (free, good coverage)
        try:
            yahoo_provider = YahooFinanceProvider()
            self.providers.append(yahoo_provider)
            logging.info("Yahoo Finance provider initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize Yahoo Finance provider: {e}")

        # Alpha Vantage (requires API key, good for forex)
        try:
            alpha_provider = AlphaVantageProvider(alpha_vantage_api_key)
            self.providers.append(alpha_provider)
            logging.info("Alpha Vantage provider initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize Alpha Vantage provider: {e}")

    def _setup_priorities(self):
        """Set up provider priorities for different instrument types."""
        self.provider_priorities = {
            InstrumentType.STOCK: ["Yahoo Finance", "Alpha Vantage"],
            InstrumentType.ETF: ["Yahoo Finance", "Alpha Vantage"],
            InstrumentType.MUTUAL_FUND: ["Yahoo Finance"],
            InstrumentType.CRYPTO: ["Yahoo Finance"],
            InstrumentType.BOND: [],  # Limited support
            InstrumentType.CASH: [],  # No provider needed
            InstrumentType.OPTION: [],  # Limited support
            InstrumentType.FUTURE: [],  # Limited support
        }

    def get_providers_for_instrument(
        self, instrument_type: InstrumentType
    ) -> List[BaseDataProvider]:
        """Get ordered list of providers that support an instrument type."""
        priority_names = self.provider_priorities.get(instrument_type, [])
        providers = []

        for name in priority_names:
            for provider in self.providers:
                if provider.name == name and provider.supports_instrument_type(
                    instrument_type
                ):
                    providers.append(provider)
                    break

        return providers

    def get_current_price(
        self, symbol: str, instrument_type: Optional[InstrumentType] = None
    ) -> Optional[Decimal]:
        """Get current price, trying providers in priority order."""
        if instrument_type:
            providers = self.get_providers_for_instrument(instrument_type)
        else:
            providers = self.providers

        for provider in providers:
            try:
                price = provider.get_current_price(symbol)
                if price is not None:
                    logging.debug(
                        f"Got current price for {symbol} from {provider.name}: {price}"
                    )
                    return price
            except Exception as e:
                logging.warning(
                    f"Error getting current price from {provider.name}: {e}"
                )
                continue

        logging.warning(f"Could not get current price for {symbol} from any provider")
        return None

    def get_historical_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        instrument_type: Optional[InstrumentType] = None,
    ) -> List[PriceData]:
        """Get historical prices, trying providers in priority order."""
        if instrument_type:
            providers = self.get_providers_for_instrument(instrument_type)
        else:
            providers = self.providers

        for provider in providers:
            try:
                prices = provider.get_historical_prices(symbol, start_date, end_date)
                if prices:
                    logging.debug(
                        f"Got {len(prices)} historical prices for {symbol} from {provider.name}"
                    )
                    return prices
            except Exception as e:
                logging.warning(
                    f"Error getting historical prices from {provider.name}: {e}"
                )
                continue

        logging.warning(
            f"Could not get historical prices for {symbol} from any provider"
        )
        return []

    def get_instrument_info(
        self, symbol: str, force_refresh: bool = False
    ) -> Optional[InstrumentInfo]:
        """Get instrument information with caching."""
        if not force_refresh and symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        for provider in self.providers:
            try:
                info = provider.get_instrument_info(symbol)
                if info:
                    logging.debug(
                        f"Got instrument info for {symbol} from {provider.name}"
                    )
                    self._instrument_cache[symbol] = info
                    return info
            except Exception as e:
                logging.warning(
                    f"Error getting instrument info from {provider.name}: {e}"
                )
                continue

        logging.warning(f"Could not get instrument info for {symbol} from any provider")
        return None

    def search_instruments(self, query: str) -> List[InstrumentInfo]:
        """Search for instruments across all providers."""
        all_results = []
        seen_symbols = set()

        for provider in self.providers:
            try:
                results = provider.search_instruments(query)
                for result in results:
                    if result.symbol not in seen_symbols:
                        all_results.append(result)
                        seen_symbols.add(result.symbol)
            except Exception as e:
                logging.warning(
                    f"Error searching instruments from {provider.name}: {e}"
                )
                continue

        return all_results[:20]  # Limit to top 20 results

    def get_exchange_rate(
        self,
        from_currency: Currency,
        to_currency: Currency,
        force_refresh: bool = False,
    ) -> Optional[Decimal]:
        """Get exchange rate with caching."""
        if from_currency == to_currency:
            return Decimal("1")

        cache_key = f"{from_currency.value}_{to_currency.value}"

        if not force_refresh and cache_key in self._exchange_rate_cache:
            return self._exchange_rate_cache[cache_key]

        # Try providers in order of forex reliability
        providers_by_fx_quality = []
        for provider in self.providers:
            if provider.name == "Alpha Vantage":
                providers_by_fx_quality.insert(
                    0, provider
                )  # Alpha Vantage first for forex
            else:
                providers_by_fx_quality.append(provider)

        for provider in providers_by_fx_quality:
            try:
                rate = provider.get_exchange_rate(from_currency, to_currency)
                if rate is not None:
                    logging.debug(
                        f"Got exchange rate {from_currency}->{to_currency} from {provider.name}: {rate}"
                    )
                    self._exchange_rate_cache[cache_key] = rate
                    return rate
            except Exception as e:
                logging.warning(
                    f"Error getting exchange rate from {provider.name}: {e}"
                )
                continue

        logging.warning(
            f"Could not get exchange rate {from_currency} to {to_currency} from any provider"
        )
        return None

    def get_multiple_current_prices(
        self, symbols: List[str]
    ) -> Dict[str, Optional[Decimal]]:
        """Get current prices for multiple symbols efficiently."""
        results = {}

        for symbol in symbols:
            results[symbol] = self.get_current_price(symbol)

        return results

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists across all providers."""
        for provider in self.providers:
            try:
                if provider.validate_symbol(symbol):
                    return True
            except Exception:
                continue

        return False

    def clear_cache(self):
        """Clear all cached data."""
        self._instrument_cache.clear()
        self._exchange_rate_cache.clear()
        logging.info("Data provider cache cleared")

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all providers."""
        status = {}

        for provider in self.providers:
            try:
                # Try a simple operation to test provider
                test_result = provider.get_current_price(
                    "AAPL"
                )  # Test with Apple stock
                status[provider.name] = test_result is not None
            except Exception:
                status[provider.name] = False

        return status
