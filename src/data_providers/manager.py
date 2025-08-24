"""
Data provider manager for coordinating multiple financial data sources.
"""

import logging
import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from ..portfolio.models import Currency, InstrumentType

from .base import BaseDataProvider, InstrumentInfo, PriceData
from .yahoo_finance import YahooFinanceProvider


class DataProviderManager:
    """Manages multiple data providers and routes requests appropriately."""

    def __init__(self):
        """Initialize the data provider manager."""
        self.providers: List[BaseDataProvider] = []
        self.provider_priorities: Dict[InstrumentType, List[str]] = {}

        # Initialize providers
        self._setup_providers()
        self._setup_priorities()

        # Cache for instrument info to avoid repeated API calls
        self._instrument_cache: Dict[str, tuple[InstrumentInfo, float]] = {}
        self._exchange_rate_cache: Dict[str, tuple[Decimal, float]] = {}
        self._failed_symbols_cache: Dict[str, float] = {}
        self._failed_isins_cache: Dict[str, float] = {}
        self._positive_cache_ttl = 3600  # 1 hour for successful lookups
        self._negative_cache_ttl = 86400  # 24 hours for failed lookups

    def _setup_providers(self):
        """Set up available data providers."""
        # Yahoo Finance (free, good coverage)
        try:
            yahoo_provider = YahooFinanceProvider()
            self.providers.append(yahoo_provider)
            logging.info("Yahoo Finance provider initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize Yahoo Finance provider: {e}")

    def _setup_priorities(self):
        """Set up provider priorities for different instrument types."""
        self.provider_priorities = {
            InstrumentType.STOCK: ["Yahoo Finance"],
            InstrumentType.ETF: ["Yahoo Finance"],
            InstrumentType.MUTUAL_FUND: ["Yahoo Finance"],
            InstrumentType.CRYPTO: ["Yahoo Finance"],
            InstrumentType.BOND: ["Yahoo Finance"],  # Bond ETFs and some individual bonds
            InstrumentType.CASH: [],  # No provider needed
            InstrumentType.OPTION: [],  # Limited support
            InstrumentType.FUTURE: [],  # Limited support
        }

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize user-provided symbols to provider-friendly format.

        - Strip leading '$' often used in chats (e.g., $GOOGL -> GOOGL)
        - Trim whitespace and uppercase
        """
        if not symbol:
            return symbol
        return symbol.strip().lstrip("$").upper()

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
        symbol = self._normalize_symbol(symbol)
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
        symbol = self._normalize_symbol(symbol)
        if instrument_type:
            providers = self.get_providers_for_instrument(instrument_type)
        else:
            providers = self.providers

        # First attempt: requested window
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

        # Fallback for single-day windows: short lookaround to handle weekends/holidays
        if start_date == end_date:
            fallback_start = start_date - timedelta(days=3)
            fallback_end = end_date + timedelta(days=3)
            for provider in providers:
                try:
                    prices = provider.get_historical_prices(
                        symbol, fallback_start, fallback_end
                    )
                    if prices:
                        logging.debug(
                            f"Got {len(prices)} historical prices for {symbol} from {provider.name} using fallback window {fallback_start}..{fallback_end}"
                        )
                        return prices
                except Exception as e:
                    logging.warning(
                        f"Error getting fallback historical prices from {provider.name}: {e}"
                    )
                    continue

        logging.info(
            f"No historical prices for {symbol} in {start_date}..{end_date}. Market may have been closed or symbol unavailable."
        )
        return []

    def get_instrument_info(
        self, symbol: str, force_refresh: bool = False
    ) -> Optional[InstrumentInfo]:
        """Get instrument information with caching."""
        symbol = self._normalize_symbol(symbol)

        # Check negative cache first
        if not force_refresh and symbol in self._failed_symbols_cache:
            cache_time = self._failed_symbols_cache[symbol]
            if time.time() - cache_time < self._negative_cache_ttl:
                return None

        # Check positive cache
        if not force_refresh and symbol in self._instrument_cache:
            info, cache_time = self._instrument_cache[symbol]
            if time.time() - cache_time < self._positive_cache_ttl:
                return info

        for provider in self.providers:
            try:
                info = provider.get_instrument_info(symbol)
                if info:
                    logging.debug(
                        f"Got instrument info for {symbol} from {provider.name}"
                    )
                    self._instrument_cache[symbol] = (info, time.time())
                    return info
            except Exception as e:
                logging.warning(
                    f"Error getting instrument info from {provider.name}: {e}"
                )
                continue

        # Cache negative result
        self._failed_symbols_cache[symbol] = time.time()
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

    def get_historical_fx_rate_on(
        self, day: date, from_currency: Currency, to_currency: Currency
    ) -> Optional[Decimal]:
        """Get historical FX close for a specific day using Yahoo symbol pairs.

        Attempts direct pair {FROM}{TO}=X; if unavailable, tries inverse and inverts the rate.
        Looks up the exact date; if missing (holiday/weekend), looks ahead up to 3 days for the first available.
        """
        if from_currency == to_currency:
            return Decimal("1")

        pair = f"{from_currency.value}{to_currency.value}=X"
        inverse_pair = f"{to_currency.value}{from_currency.value}=X"

        # Prefer Yahoo provider for historical series
        yahoo_provider: Optional[BaseDataProvider] = None
        for p in self.providers:
            if p.name == "Yahoo Finance":
                yahoo_provider = p
                break

        if not yahoo_provider:
            return None

        def _extract_close(symbol: str, start: date, end: date) -> Optional[Decimal]:
            try:
                series = yahoo_provider.get_historical_prices(symbol, start, end)
                if series:
                    pd0 = series[0]
                    return (
                        pd0.close_price
                        or pd0.open_price
                        or pd0.high_price
                        or pd0.low_price
                    )
            except Exception:
                return None
            return None

        # Exact date
        rate = _extract_close(pair, day, day)
        if rate is not None:
            return rate

        # Try a short lookahead window
        lookahead_end = day + timedelta(days=3)
        rate = _extract_close(pair, day, lookahead_end)
        if rate is not None:
            return rate

        # Try inverse pair and invert
        inv = _extract_close(inverse_pair, day, day)
        if inv is None:
            inv = _extract_close(inverse_pair, day, lookahead_end)
        if inv is not None and inv != 0:
            try:
                return Decimal("1") / inv
            except Exception:
                return None

        return None

    def get_multiple_current_prices(
        self, symbols: List[str]
    ) -> Dict[str, Optional[Decimal]]:
        """Get current prices for multiple symbols efficiently."""
        results = {}

        for symbol in symbols:
            norm = self._normalize_symbol(symbol)
            results[symbol] = self.get_current_price(norm)

        return results

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists across all providers."""
        symbol = self._normalize_symbol(symbol)
        for provider in self.providers:
            try:
                if provider.validate_symbol(symbol):
                    return True
            except Exception:
                continue

        return False

    def search_by_isin(self, isin: str) -> Optional[InstrumentInfo]:
        """Search for an instrument by ISIN across all providers."""
        isin = isin.upper().strip()

        # Check negative cache first
        if isin in self._failed_isins_cache:
            cache_time = self._failed_isins_cache[isin]
            if time.time() - cache_time < self._negative_cache_ttl:
                return None

        # Check positive cache
        if isin in self._instrument_cache:
            info, cache_time = self._instrument_cache[isin]
            if time.time() - cache_time < self._positive_cache_ttl:
                return info

        # Try direct ISIN lookup first
        for provider in self.providers:
            try:
                if hasattr(provider, 'get_instrument_by_isin'):
                    info = provider.get_instrument_by_isin(isin)
                    if info:
                        self._instrument_cache[isin] = (info, time.time())
                        return info
            except Exception as e:
                logging.warning(f"Error in ISIN lookup from {provider.name}: {e}")
                continue

        # Fallback: search by ISIN as query
        try:
            search_results = self.search_instruments(isin)
            for result in search_results:
                if result.isin and result.isin.upper() == isin:
                    self._instrument_cache[isin] = (result, time.time())
                    return result
        except Exception as e:
            logging.warning(f"Error in ISIN search fallback: {e}")

        # Cache negative result
        self._failed_isins_cache[isin] = time.time()
        return None

    def validate_isin(self, isin: str) -> bool:
        """Validate if an ISIN exists using search_by_isin."""
        return self.search_by_isin(isin) is not None

    def clear_cache(self):
        """Clear all cached data."""
        self._instrument_cache.clear()
        self._exchange_rate_cache.clear()
        self._failed_symbols_cache.clear()
        self._failed_isins_cache.clear()
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
