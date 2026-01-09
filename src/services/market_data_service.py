"""
Centralized market data service with staleness tracking.

This service provides a unified interface for all market data access across
the application, with explicit staleness tracking and coordinated refresh.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set

import pandas as pd

from ..data_providers.base import InstrumentInfo, PriceData
from ..data_providers.manager import DataProviderManager
from ..portfolio.models import Currency, InstrumentType, Portfolio


@dataclass
class PriceResult:
    """Result of a price lookup with staleness information."""

    symbol: str
    price: Optional[Decimal]
    timestamp: Optional[datetime]
    is_stale: bool = False
    error: Optional[str] = None

    @property
    def age_seconds(self) -> Optional[float]:
        """Get the age of this price in seconds."""
        if self.timestamp is None:
            return None
        return (datetime.now() - self.timestamp).total_seconds()

    @property
    def age_minutes(self) -> Optional[float]:
        """Get the age of this price in minutes."""
        if self.age_seconds is None:
            return None
        return self.age_seconds / 60


@dataclass
class FXResult:
    """Result of an FX rate lookup with staleness information."""

    from_currency: Currency
    to_currency: Currency
    rate: Optional[Decimal]
    as_of_date: Optional[date]
    timestamp: Optional[datetime]
    is_stale: bool = False
    error: Optional[str] = None

    @property
    def age_seconds(self) -> Optional[float]:
        """Get the age of this rate in seconds."""
        if self.timestamp is None:
            return None
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class DataFreshness:
    """Overall data freshness status."""

    last_price_refresh: Optional[datetime] = None
    last_fx_refresh: Optional[datetime] = None
    symbols_updated: int = 0
    fx_pairs_updated: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def prices_age_minutes(self) -> Optional[float]:
        """Get the age of price data in minutes."""
        if self.last_price_refresh is None:
            return None
        return (datetime.now() - self.last_price_refresh).total_seconds() / 60

    @property
    def fx_age_minutes(self) -> Optional[float]:
        """Get the age of FX data in minutes."""
        if self.last_fx_refresh is None:
            return None
        return (datetime.now() - self.last_fx_refresh).total_seconds() / 60

    @property
    def is_stale(self) -> bool:
        """Check if data is considered stale (> 4 hours old)."""
        stale_threshold_minutes = 240  # 4 hours
        if self.prices_age_minutes is None:
            return True
        return self.prices_age_minutes > stale_threshold_minutes

    @property
    def freshness_display(self) -> str:
        """Human-readable freshness status."""
        if self.last_price_refresh is None:
            return "Never updated"

        age = self.prices_age_minutes
        if age is None:
            return "Unknown"
        elif age < 1:
            return "Just now"
        elif age < 60:
            return f"{int(age)} min ago"
        elif age < 1440:  # 24 hours
            return f"{int(age / 60)} hours ago"
        else:
            return f"{int(age / 1440)} days ago"


@dataclass
class RefreshResult:
    """Result of a data refresh operation."""

    success: bool
    symbols_updated: int = 0
    fx_pairs_updated: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class MarketDataService:
    """Centralized market data access with staleness tracking.

    This service wraps DataProviderManager and adds:
    - Explicit staleness tracking for all data
    - Coordinated refresh across all symbols
    - DataFrame-based price history for analytics
    - Observable freshness state for UI
    """

    # Staleness threshold in seconds (1 hour)
    PRICE_STALE_THRESHOLD = 3600
    FX_STALE_THRESHOLD = 3600

    def __init__(self, data_manager: Optional[DataProviderManager] = None):
        """Initialize the market data service.

        Args:
            data_manager: Optional DataProviderManager instance. If not provided,
                         a new one will be created.
        """
        self._data_manager = data_manager or DataProviderManager()

        # Price cache with timestamps: symbol -> PriceResult
        self._price_cache: Dict[str, PriceResult] = {}

        # FX cache with timestamps: "FROM_TO" -> FXResult
        self._fx_cache: Dict[str, FXResult] = {}

        # Track overall freshness
        self._freshness = DataFreshness()

        # Track symbols we're managing
        self._tracked_symbols: Set[str] = set()

    @property
    def data_manager(self) -> DataProviderManager:
        """Access to underlying DataProviderManager for compatibility."""
        return self._data_manager

    @property
    def freshness(self) -> DataFreshness:
        """Get current data freshness status."""
        return self._freshness

    def get_current_price(
        self,
        symbol: str,
        instrument_type: Optional[InstrumentType] = None,
        force_refresh: bool = False,
    ) -> PriceResult:
        """Get current price with staleness information.

        Args:
            symbol: The trading symbol
            instrument_type: Optional instrument type for provider selection
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            PriceResult with price and staleness info
        """
        symbol = symbol.upper().strip()

        # Check cache unless forcing refresh
        if not force_refresh and symbol in self._price_cache:
            cached = self._price_cache[symbol]
            # Update staleness flag based on current time
            if cached.timestamp:
                age = (datetime.now() - cached.timestamp).total_seconds()
                cached.is_stale = age > self.PRICE_STALE_THRESHOLD
            return cached

        # Fetch from provider
        try:
            price = self._data_manager.get_current_price(symbol, instrument_type)
            now = datetime.now()

            result = PriceResult(
                symbol=symbol,
                price=price,
                timestamp=now if price is not None else None,
                is_stale=False,
                error=None if price is not None else f"No price available for {symbol}",
            )

            # Cache the result
            self._price_cache[symbol] = result
            self._tracked_symbols.add(symbol)

            # Update overall freshness when we successfully fetch a price
            if price is not None:
                self._freshness.last_price_refresh = now
                self._freshness.symbols_updated = len(
                    [p for p in self._price_cache.values() if p.price is not None]
                )

            return result

        except Exception as e:
            logging.error(f"Error fetching price for {symbol}: {e}")
            return PriceResult(
                symbol=symbol,
                price=None,
                timestamp=None,
                is_stale=True,
                error=str(e),
            )

    def get_current_prices(
        self,
        symbols: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, PriceResult]:
        """Get current prices for multiple symbols.

        Args:
            symbols: List of trading symbols
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            Dict mapping symbol to PriceResult
        """
        results: Dict[str, PriceResult] = {}

        for symbol in symbols:
            results[symbol] = self.get_current_price(symbol, force_refresh=force_refresh)

        return results

    def get_fx_rate(
        self,
        from_currency: Currency,
        to_currency: Currency,
        as_of: Optional[date] = None,
        force_refresh: bool = False,
    ) -> FXResult:
        """Get exchange rate with staleness information.

        Args:
            from_currency: Source currency
            to_currency: Target currency
            as_of: Optional historical date. If None, gets current rate.
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            FXResult with rate and staleness info
        """
        if from_currency == to_currency:
            return FXResult(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=Decimal("1"),
                as_of_date=as_of or date.today(),
                timestamp=datetime.now(),
                is_stale=False,
            )

        cache_key = f"{from_currency.value}_{to_currency.value}"
        if as_of:
            cache_key += f"_{as_of.isoformat()}"

        # Check cache unless forcing refresh
        if not force_refresh and cache_key in self._fx_cache:
            cached = self._fx_cache[cache_key]
            # Update staleness for current rates only
            if cached.timestamp and as_of is None:
                age = (datetime.now() - cached.timestamp).total_seconds()
                cached.is_stale = age > self.FX_STALE_THRESHOLD
            return cached

        # Fetch from provider
        try:
            if as_of:
                rate = self._data_manager.get_historical_fx_rate_on(
                    as_of, from_currency, to_currency
                )
            else:
                rate = self._data_manager.get_exchange_rate(from_currency, to_currency)

            now = datetime.now()

            result = FXResult(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=rate,
                as_of_date=as_of or date.today(),
                timestamp=now if rate is not None else None,
                is_stale=False,
                error=None if rate is not None else f"No rate for {from_currency}->{to_currency}",
            )

            # Cache the result
            self._fx_cache[cache_key] = result

            # Update overall freshness when we successfully fetch an FX rate
            if rate is not None and as_of is None:  # Only for current rates
                self._freshness.last_fx_refresh = now
                self._freshness.fx_pairs_updated = len(
                    [f for f in self._fx_cache.values() if f.rate is not None]
                )

            return result

        except Exception as e:
            logging.error(f"Error fetching FX rate {from_currency}->{to_currency}: {e}")
            return FXResult(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=None,
                as_of_date=as_of,
                timestamp=None,
                is_stale=True,
                error=str(e),
            )

    def get_price_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        instrument_type: Optional[InstrumentType] = None,
    ) -> pd.DataFrame:
        """Get historical prices as a DataFrame.

        Args:
            symbol: The trading symbol
            start_date: Start date for history
            end_date: End date for history
            instrument_type: Optional instrument type for provider selection

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Index is the date column.
        """
        symbol = symbol.upper().strip()

        try:
            prices: List[PriceData] = self._data_manager.get_historical_prices(
                symbol, start_date, end_date, instrument_type
            )

            if not prices:
                # Return empty DataFrame with correct schema
                return pd.DataFrame(
                    columns=["date", "open", "high", "low", "close", "volume"]
                ).set_index("date")

            # Convert to DataFrame
            data = []
            for p in prices:
                data.append({
                    "date": p.date,
                    "open": float(p.open_price) if p.open_price else None,
                    "high": float(p.high_price) if p.high_price else None,
                    "low": float(p.low_price) if p.low_price else None,
                    "close": float(p.close_price) if p.close_price else None,
                    "volume": p.volume,
                })

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            return df

        except Exception as e:
            logging.error(f"Error fetching price history for {symbol}: {e}")
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            ).set_index("date")

    def get_portfolio_prices_df(
        self,
        portfolio: Portfolio,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Get historical prices for all portfolio positions as a DataFrame.

        Args:
            portfolio: The portfolio to get prices for
            start_date: Start date for history
            end_date: End date for history

        Returns:
            DataFrame with date index and one column per symbol (close prices).
            Missing dates are forward-filled.
        """
        symbols = list(portfolio.positions.keys())
        if not symbols:
            return pd.DataFrame()

        # Fetch price history for each symbol
        price_series: Dict[str, pd.Series] = {}
        for symbol in symbols:
            position = portfolio.positions[symbol]
            df = self.get_price_history(
                symbol,
                start_date,
                end_date,
                position.instrument.instrument_type,
            )
            if not df.empty and "close" in df.columns:
                price_series[symbol] = df["close"]

        if not price_series:
            return pd.DataFrame()

        # Combine into single DataFrame
        combined = pd.DataFrame(price_series)

        # Create complete date range and reindex
        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        combined = combined.reindex(all_dates)

        # Forward-fill missing values
        combined = combined.ffill()

        return combined

    def refresh_prices(self, symbols: List[str]) -> RefreshResult:
        """Refresh prices for specified symbols.

        Args:
            symbols: List of symbols to refresh

        Returns:
            RefreshResult with update stats
        """
        import time

        start_time = time.time()
        updated = 0
        errors: List[str] = []

        for symbol in symbols:
            result = self.get_current_price(symbol, force_refresh=True)
            if result.price is not None:
                updated += 1
            elif result.error:
                errors.append(result.error)

        duration = time.time() - start_time

        # Update freshness
        self._freshness.last_price_refresh = datetime.now()
        self._freshness.symbols_updated = updated
        self._freshness.errors = errors

        return RefreshResult(
            success=len(errors) == 0,
            symbols_updated=updated,
            errors=errors,
            duration_seconds=duration,
        )

    def refresh_all(self, portfolio: Optional[Portfolio] = None) -> RefreshResult:
        """Refresh all tracked data, optionally for a specific portfolio.

        Args:
            portfolio: Optional portfolio to refresh data for

        Returns:
            RefreshResult with update stats
        """
        import time

        start_time = time.time()
        errors: List[str] = []

        # Determine which symbols to refresh
        if portfolio:
            symbols = list(portfolio.positions.keys())
        else:
            symbols = list(self._tracked_symbols)

        # Refresh prices
        price_result = self.refresh_prices(symbols)
        errors.extend(price_result.errors)

        # Refresh FX rates for portfolio currencies if provided
        fx_updated = 0
        if portfolio:
            currencies = set()
            currencies.add(portfolio.base_currency)
            for currency in portfolio.cash_balances.keys():
                currencies.add(currency)
            for position in portfolio.positions.values():
                currencies.add(position.instrument.currency)

            # Refresh rates to base currency
            for currency in currencies:
                if currency != portfolio.base_currency:
                    result = self.get_fx_rate(
                        currency, portfolio.base_currency, force_refresh=True
                    )
                    if result.rate is not None:
                        fx_updated += 1
                    elif result.error:
                        errors.append(result.error)

        duration = time.time() - start_time

        # Update freshness
        self._freshness.last_price_refresh = datetime.now()
        self._freshness.last_fx_refresh = datetime.now()
        self._freshness.symbols_updated = price_result.symbols_updated
        self._freshness.fx_pairs_updated = fx_updated
        self._freshness.errors = errors

        return RefreshResult(
            success=len(errors) == 0,
            symbols_updated=price_result.symbols_updated,
            fx_pairs_updated=fx_updated,
            errors=errors,
            duration_seconds=duration,
        )

    def get_instrument_info(
        self, symbol: str, force_refresh: bool = False
    ) -> Optional[InstrumentInfo]:
        """Get instrument information.

        Args:
            symbol: The trading symbol
            force_refresh: If True, bypass cache

        Returns:
            InstrumentInfo or None if not found
        """
        return self._data_manager.get_instrument_info(symbol, force_refresh)

    def search_instruments(self, query: str) -> List[InstrumentInfo]:
        """Search for instruments by name or symbol.

        Args:
            query: Search query

        Returns:
            List of matching instruments
        """
        return self._data_manager.search_instruments(query)

    def search_by_isin(self, isin: str) -> Optional[InstrumentInfo]:
        """Search for instrument by ISIN.

        Args:
            isin: ISIN to search for

        Returns:
            InstrumentInfo or None if not found
        """
        return self._data_manager.search_by_isin(isin)

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists.

        Args:
            symbol: The trading symbol to validate

        Returns:
            True if symbol is valid
        """
        return self._data_manager.validate_symbol(symbol)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._price_cache.clear()
        self._fx_cache.clear()
        self._data_manager.clear_cache()
        self._freshness = DataFreshness()
        logging.info("MarketDataService cache cleared")

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all data providers.

        Returns:
            Dict mapping provider name to availability status
        """
        return self._data_manager.get_provider_status()
