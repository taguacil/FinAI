"""
Base interface for financial data providers.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from functools import wraps
from typing import Callable, List, Optional

from ..portfolio.models import Currency, InstrumentType


def retry_with_backoff(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, ConnectionError, TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise e

                    delay = min(base_delay * (2**attempt), max_delay)
                    logging.warning(
                        f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                except Exception as e:
                    # Don't retry on other exceptions
                    raise e
            return None

        return wrapper

    return decorator


@dataclass
class PriceData:
    """Represents price data for a financial instrument."""

    symbol: str
    date: date
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None
    close_price: Optional[Decimal] = None
    volume: Optional[int] = None
    currency: Optional[Currency] = None


@dataclass
class InstrumentInfo:
    """Represents detailed information about a financial instrument."""

    symbol: str
    name: str
    instrument_type: InstrumentType
    currency: Currency
    exchange: Optional[str] = None
    isin: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[Decimal] = None
    description: Optional[str] = None


class BaseDataProvider(ABC):
    """Abstract base class for all financial data providers."""

    def __init__(self):
        self.name = "Base Provider"
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum seconds between requests

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    @retry_with_backoff(max_retries=3)
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a financial instrument."""
        pass

    @retry_with_backoff(max_retries=3)
    @abstractmethod
    def get_historical_prices(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[PriceData]:
        """Get historical price data for a financial instrument."""
        pass

    @retry_with_backoff(max_retries=3)
    @abstractmethod
    def get_instrument_info(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get detailed information about a financial instrument."""
        pass

    @retry_with_backoff(max_retries=2)
    @abstractmethod
    def search_instruments(self, query: str) -> List[InstrumentInfo]:
        """Search for financial instruments by name or symbol."""
        pass

    @retry_with_backoff(max_retries=3)
    @abstractmethod
    def get_exchange_rate(
        self, from_currency: Currency, to_currency: Currency
    ) -> Optional[Decimal]:
        """Get current exchange rate between two currencies."""
        pass

    @abstractmethod
    def supports_instrument_type(self, instrument_type: InstrumentType) -> bool:
        """Check if this provider supports a specific instrument type."""
        pass

    def validate_symbol(self, symbol: str) -> bool:
        """Validate a trading symbol."""
        try:
            info = self.get_instrument_info(symbol)
            return info is not None
        except Exception:
            return False


class DataProviderError(Exception):
    """Base exception for data provider errors."""

    pass


class RateLimitError(DataProviderError):
    """Raised when rate limit is exceeded."""

    pass


class InvalidSymbolError(DataProviderError):
    """Raised when an invalid symbol is requested."""

    pass


class ConnectionError(DataProviderError):
    """Raised when connection to data provider fails."""

    pass


class TimeoutError(DataProviderError):
    """Raised when request times out."""

    pass
