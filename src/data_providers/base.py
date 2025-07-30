"""
Base interface for financial data providers.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..portfolio.models import Currency, InstrumentType


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
    """Abstract base class for financial data providers."""
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol."""
        pass
    
    @abstractmethod
    def get_historical_prices(self, symbol: str, start_date: date, end_date: date) -> List[PriceData]:
        """Get historical price data for a symbol."""
        pass
    
    @abstractmethod
    def get_instrument_info(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get detailed information about an instrument."""
        pass
    
    @abstractmethod
    def search_instruments(self, query: str) -> List[InstrumentInfo]:
        """Search for instruments by name or symbol."""
        pass
    
    @abstractmethod
    def get_exchange_rate(self, from_currency: Currency, to_currency: Currency) -> Optional[Decimal]:
        """Get current exchange rate between two currencies."""
        pass
    
    @abstractmethod
    def supports_instrument_type(self, instrument_type: InstrumentType) -> bool:
        """Check if provider supports a specific instrument type."""
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists and is tradeable."""
        try:
            info = self.get_instrument_info(symbol)
            return info is not None
        except Exception:
            return False


class DataProviderError(Exception):
    """Custom exception for data provider errors."""
    pass


class RateLimitError(DataProviderError):
    """Exception for rate limit exceeded errors."""
    pass


class InvalidSymbolError(DataProviderError):
    """Exception for invalid symbol errors."""
    pass