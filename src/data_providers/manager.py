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
from .fx_cache import FXRateCache
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

        # Initialize persistent FX rate cache
        self.fx_cache = FXRateCache()

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
            InstrumentType.BOND: [
                "Yahoo Finance"
            ],  # Bond ETFs and some individual bonds
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
        """Get exchange rate with persistent caching."""
        if from_currency == to_currency:
            return Decimal("1")

        # Check persistent cache first (unless force refresh)
        if not force_refresh:
            # Try current rate from persistent cache
            cached_rate = self.fx_cache.get_current_rate(from_currency, to_currency)
            if cached_rate is not None:
                logging.debug(f"Got exchange rate {from_currency}->{to_currency} from cache: {cached_rate}")
                return cached_rate

        # Check in-memory cache
        cache_key = f"{from_currency.value}_{to_currency.value}"
        if not force_refresh and cache_key in self._exchange_rate_cache:
            rate, cached_time = self._exchange_rate_cache[cache_key]
            # Check if cache is still fresh (1 hour)
            if time.time() - cached_time < self._positive_cache_ttl:
                return rate

        # Fetch from providers
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

                    # Store in both in-memory and persistent cache
                    self._exchange_rate_cache[cache_key] = (rate, time.time())
                    self.fx_cache.store_rate(from_currency, to_currency, date.today(), rate)

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
        """Get historical FX close for a specific day using persistent caching.

        Attempts direct pair {FROM}{TO}=X; if unavailable, tries inverse and inverts the rate.
        Looks up the exact date; if missing (holiday/weekend), looks ahead up to 3 days for the first available.
        """
        if from_currency == to_currency:
            return Decimal("1")

        # Check persistent cache first
        cached_rate = self.fx_cache.get_rate(from_currency, to_currency, day)
        if cached_rate is not None:
            logging.debug(f"Got historical FX rate {from_currency}->{to_currency} for {day} from cache: {cached_rate}")
            return cached_rate

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
            # Store in cache for future use
            self.fx_cache.store_rate(from_currency, to_currency, day, rate)
            return rate

        # Try a short lookahead window
        lookahead_end = day + timedelta(days=3)
        rate = _extract_close(pair, day, lookahead_end)
        if rate is not None:
            # Store in cache for future use
            self.fx_cache.store_rate(from_currency, to_currency, day, rate)
            return rate

        # Try inverse pair and invert
        inv = _extract_close(inverse_pair, day, day)
        if inv is None:
            inv = _extract_close(inverse_pair, day, lookahead_end)
        if inv is not None and inv != 0:
            try:
                inverted_rate = Decimal("1") / inv
                # Store the original pair rate in cache
                self.fx_cache.store_rate(from_currency, to_currency, day, inverted_rate)
                return inverted_rate
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
        """Search for an instrument by ISIN from known mappings.

        Note: ISIN is always provided by the user, so we focus on known mappings
        rather than searching external sources.
        """
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

        # Try direct ISIN lookup from providers first
        for provider in self.providers:
            try:
                if hasattr(provider, "get_instrument_by_isin"):
                    info = provider.get_instrument_by_isin(isin)
                    if info:
                        self._instrument_cache[isin] = (info, time.time())
                        return info
            except Exception as e:
                logging.warning(f"Error in ISIN lookup from {provider.name}: {e}")
                continue

        # Use known ISIN mappings for common instruments
        known_isins = self._get_known_isin_mappings()
        if isin in known_isins:
            known_info = known_isins[isin]
            # Create InstrumentInfo from known data
            from .base import InstrumentInfo

            instrument_info = InstrumentInfo(
                symbol=known_info["symbol"],
                name=known_info["name"],
                instrument_type=known_info["type"],
                currency=known_info["currency"],
                exchange=known_info.get("exchange"),
                isin=isin,
                sector=known_info.get("sector"),
                industry=known_info.get("industry"),
            )
            self._instrument_cache[isin] = (instrument_info, time.time())
            return instrument_info

        # Cache negative result
        self._failed_isins_cache[isin] = time.time()
        return None

    def _get_known_isin_mappings(self) -> Dict[str, Dict]:
        """Get known ISIN to instrument mappings for common instruments."""
        from ..portfolio.models import Currency, InstrumentType

        return {
            # Apple Inc.
            "US0378331005": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Consumer Electronics",
            },
            # Microsoft Corporation
            "US5949181045": {
                "symbol": "MSFT",
                "name": "Microsoft Corporation",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Software",
            },
            # Alphabet Inc. (Google)
            "US02079K3059": {
                "symbol": "GOOGL",
                "name": "Alphabet Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Internet Services",
            },
            # Tesla Inc.
            "US88160R1014": {
                "symbol": "TSLA",
                "name": "Tesla Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Consumer Discretionary",
                "industry": "Automobiles",
            },
            # Amazon.com Inc.
            "US0231351067": {
                "symbol": "AMZN",
                "name": "Amazon.com Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Consumer Discretionary",
                "industry": "Internet Retail",
            },
            # Meta Platforms Inc. (Facebook)
            "US30303M1027": {
                "symbol": "META",
                "name": "Meta Platforms Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Internet Services",
            },
            # NVIDIA Corporation
            "US67066G1040": {
                "symbol": "NVDA",
                "name": "NVIDIA Corporation",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Semiconductors",
            },
            # Netflix Inc.
            "US64110L1061": {
                "symbol": "NFLX",
                "name": "Netflix Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Communication Services",
                "industry": "Entertainment",
            },
            # Berkshire Hathaway Inc. Class A
            "US0846707026": {
                "symbol": "BRK.A",
                "name": "Berkshire Hathaway Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Financial Services",
                "industry": "Insurance",
            },
            # Berkshire Hathaway Inc. Class B
            "US0846701086": {
                "symbol": "BRK.B",
                "name": "Berkshire Hathaway Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Financial Services",
                "industry": "Insurance",
            },
            # JPMorgan Chase & Co.
            "US46625H1005": {
                "symbol": "JPM",
                "name": "JPMorgan Chase & Co.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Financial Services",
                "industry": "Banks",
            },
            # Johnson & Johnson
            "US4781601046": {
                "symbol": "JNJ",
                "name": "Johnson & Johnson",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Healthcare",
                "industry": "Drug Manufacturers",
            },
            # Visa Inc.
            "US92826C8394": {
                "symbol": "V",
                "name": "Visa Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Financial Services",
                "industry": "Credit Services",
            },
            # Procter & Gamble Co.
            "US7427181091": {
                "symbol": "PG",
                "name": "The Procter & Gamble Co.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Consumer Defensive",
                "industry": "Household & Personal Products",
            },
            # UnitedHealth Group Inc.
            "US91324P1021": {
                "symbol": "UNH",
                "name": "UnitedHealth Group Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Healthcare",
                "industry": "Healthcare Plans",
            },
            # The Home Depot Inc.
            "US4370761029": {
                "symbol": "HD",
                "name": "The Home Depot Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Consumer Discretionary",
                "industry": "Home Improvement Retail",
            },
            # Mastercard Inc.
            "US57636Q1040": {
                "symbol": "MA",
                "name": "Mastercard Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Financial Services",
                "industry": "Credit Services",
            },
            # The Walt Disney Company
            "US2546871060": {
                "symbol": "DIS",
                "name": "The Walt Disney Company",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NYSE",
                "sector": "Communication Services",
                "industry": "Entertainment",
            },
            # PayPal Holdings Inc.
            "US70450Y1038": {
                "symbol": "PYPL",
                "name": "PayPal Holdings Inc.",
                "type": InstrumentType.STOCK,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Financial Services",
                "industry": "Credit Services",
            },
            # Common Bond ETFs
            # iShares 20+ Year Treasury Bond ETF
            "US4642876555": {
                "symbol": "TLT",
                "name": "iShares 20+ Year Treasury Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Government Bonds",
            },
            # iShares 7-10 Year Treasury Bond ETF
            "US4642872008": {
                "symbol": "IEF",
                "name": "iShares 7-10 Year Treasury Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Government Bonds",
            },
            # iShares TIPS Bond ETF
            "US4642872065": {
                "symbol": "TIP",
                "name": "iShares TIPS Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Government Bonds",
            },
            # iShares iBoxx $ Investment Grade Corporate Bond ETF
            "US4642872172": {
                "symbol": "LQD",
                "name": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Corporate Bonds",
            },
            # iShares iBoxx $ High Yield Corporate Bond ETF
            "US4642872347": {
                "symbol": "HYG",
                "name": "iShares iBoxx $ High Yield Corporate Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "High Yield Bonds",
            },
            # iShares J.P. Morgan USD Emerging Markets Bond ETF
            "US4642872354": {
                "symbol": "EMB",
                "name": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Emerging Market Bonds",
            },
            # iShares 1-3 Year Treasury Bond ETF
            "US4642872065": {
                "symbol": "SHY",
                "name": "iShares 1-3 Year Treasury Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Government Bonds",
            },
            # iShares Short Treasury Bond ETF
            "US4642872172": {
                "symbol": "SHV",
                "name": "iShares Short Treasury Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Government Bonds",
            },
            # Vanguard Total Bond Market ETF
            "US9229083636": {
                "symbol": "BND",
                "name": "Vanguard Total Bond Market ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Broad Market Bonds",
            },
            # iShares Core U.S. Aggregate Bond ETF
            "US4642872347": {
                "symbol": "AGG",
                "name": "iShares Core U.S. Aggregate Bond ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Broad Market Bonds",
            },
            # SPDR Bloomberg 1-3 Month T-Bill ETF
            "US78464A7353": {
                "symbol": "BIL",
                "name": "SPDR Bloomberg 1-3 Month T-Bill ETF",
                "type": InstrumentType.BOND,
                "currency": Currency.USD,
                "exchange": "NASDAQ",
                "sector": "Fixed Income",
                "industry": "Government Bills",
            },
        }

    def search_by_company_name(self, company_name: str) -> List[InstrumentInfo]:
        """Search for instruments by company name across all providers."""
        company_name = company_name.strip()
        if not company_name:
            return []

        all_results = []
        seen_symbols = set()

        # Try searching with the company name
        for provider in self.providers:
            try:
                results = provider.search_instruments(company_name)
                for result in results:
                    if result.symbol not in seen_symbols:
                        all_results.append(result)
                        seen_symbols.add(result.symbol)
            except Exception as e:
                logging.warning(
                    f"Error searching by company name from {provider.name}: {e}"
                )
                continue

        # If no results, try with common variations
        if not all_results:
            variations = [
                company_name + " stock",
                company_name + " shares",
                company_name + " inc",
                company_name + " corp",
                company_name + " ltd",
                company_name + " company",
            ]

            for variation in variations:
                for provider in self.providers:
                    try:
                        results = provider.search_instruments(variation)
                        for result in results:
                            if result.symbol not in seen_symbols:
                                all_results.append(result)
                                seen_symbols.add(result.symbol)
                    except Exception:
                        continue
                    if all_results:  # Stop if we found something
                        break
                if all_results:  # Stop if we found something
                    break

        return all_results[:20]  # Limit to top 20 results

    def validate_isin(self, isin: str) -> bool:
        """Validate if an ISIN exists using search_by_isin."""
        return self.search_by_isin(isin) is not None

    def clear_cache(self):
        """Clear all cached data."""
        self._instrument_cache.clear()
        self._exchange_rate_cache.clear()
        self._failed_symbols_cache.clear()
        self._failed_isins_cache.clear()
        self.fx_cache.clear_cache()
        logging.info("Data provider cache cleared")

    def get_fx_cache_stats(self) -> Dict[str, int]:
        """Get FX cache statistics."""
        return self.fx_cache.get_cache_stats()

    def cleanup_old_fx_rates(self, days_to_keep: int = 365):
        """Remove old FX rates from cache."""
        self.fx_cache.cleanup_old_rates(days_to_keep)
        logging.info(f"Cleaned up FX rates older than {days_to_keep} days")

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
