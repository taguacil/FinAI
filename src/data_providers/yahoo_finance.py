"""
Yahoo Finance data provider implementation.
"""

import time
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional

import pandas as pd
import yfinance as yf

from ..portfolio.models import Currency, InstrumentType
from .base import (
    BaseDataProvider,
    ConnectionError,
    InstrumentInfo,
    InvalidSymbolError,
    PriceData,
    RateLimitError,
    TimeoutError,
)


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider for stocks, ETFs, and some other instruments."""

    def __init__(self):
        """Initialize Yahoo Finance provider."""
        self.name = "Yahoo Finance"
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum time between requests (100ms)

    def _rate_limit(self):
        """Simple rate limiting to avoid overwhelming the API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get yfinance Ticker object with rate limiting."""
        self._rate_limit()
        # Normalize common user inputs like $SYMBOL
        norm = symbol.strip().lstrip("$")
        return yf.Ticker(norm)

    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol."""
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info

            if not info or info.get("regularMarketPrice") is None:
                raise InvalidSymbolError(f"Invalid or delisted symbol: {symbol}")

            # Try different price fields
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            )

            if price is not None:
                return Decimal(str(price))
            return None

        except InvalidSymbolError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg:
                raise RateLimitError(f"Rate limit exceeded for Yahoo Finance: {e}")
            elif "connection" in error_msg or "network" in error_msg:
                raise ConnectionError(f"Network error accessing Yahoo Finance: {e}")
            elif "timeout" in error_msg:
                raise TimeoutError(f"Timeout accessing Yahoo Finance: {e}")
            else:
                print(f"Error getting current price for {symbol}: {e}")
                return None

    def get_historical_prices(
        self, symbol: str, start_date: date, end_date: date
    ) -> List[PriceData]:
        """Get historical price data for a symbol."""
        try:
            ticker = self._get_ticker(symbol)

            # Get historical data
            hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))

            if hist.empty:
                return []

            price_data = []
            for date_idx, row in hist.iterrows():
                try:
                    price_data.append(
                        PriceData(
                            symbol=symbol,
                            date=date_idx.date(),
                            open_price=(
                                Decimal(str(row["Open"]))
                                if not pd.isna(row["Open"])
                                else None
                            ),
                            high_price=(
                                Decimal(str(row["High"]))
                                if not pd.isna(row["High"])
                                else None
                            ),
                            low_price=(
                                Decimal(str(row["Low"]))
                                if not pd.isna(row["Low"])
                                else None
                            ),
                            close_price=(
                                Decimal(str(row["Close"]))
                                if not pd.isna(row["Close"])
                                else None
                            ),
                            volume=(
                                int(row["Volume"])
                                if not pd.isna(row["Volume"])
                                else None
                            ),
                        )
                    )
                except (ValueError, TypeError) as e:
                    print(
                        f"Error processing price data for {symbol} on {date_idx.date()}: {e}"
                    )
                    continue

            return price_data

        except Exception as e:
            print(f"Error getting historical prices for {symbol}: {e}")
            return []

    def get_instrument_info(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get detailed information about an instrument."""
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info

            if not info or "symbol" not in info:
                return None

            # Map Yahoo Finance quote types to our instrument types
            quote_type = info.get("quoteType", "").upper()
            name = (info.get("longName") or info.get("shortName") or "").lower()
            symbol = symbol.upper()

            # Bond detection logic
            if (quote_type == "ETF" and any(bond_hint in name for bond_hint in
                ["bond", "treasury", "government", "corporate", "municipal", "t-bill", "tbill"])):
                instrument_type = InstrumentType.BOND
            elif (quote_type == "ETF" and any(bond_hint in symbol for bond_hint in
                ["BIL", "TLT", "IEF", "TIP", "LQD", "HYG", "EMB", "SHY", "SHV", "BND", "AGG"])):
                instrument_type = InstrumentType.BOND
            elif quote_type in ["EQUITY", "STOCK"]:
                instrument_type = InstrumentType.STOCK
            elif quote_type == "ETF":
                instrument_type = InstrumentType.ETF
            elif quote_type == "MUTUALFUND":
                instrument_type = InstrumentType.MUTUAL_FUND
            elif quote_type == "CRYPTOCURRENCY":
                instrument_type = InstrumentType.CRYPTO
            else:
                instrument_type = InstrumentType.STOCK  # Default fallback

            # Get currency
            currency_str = info.get("currency", "USD")
            try:
                currency = Currency(currency_str)
            except ValueError:
                currency = Currency.USD

            return InstrumentInfo(
                symbol=symbol.upper(),
                name=info.get("longName") or info.get("shortName") or symbol,
                instrument_type=instrument_type,
                currency=currency,
                exchange=info.get("exchange"),
                isin=info.get("isin"),
                sector=info.get("sector"),
                industry=info.get("industry"),
                market_cap=(
                    Decimal(str(info["marketCap"])) if info.get("marketCap") else None
                ),
                description=info.get("longBusinessSummary"),
            )

        except Exception as e:
            print(f"Error getting instrument info for {symbol}: {e}")
            return None

    def search_instruments(self, query: str) -> List[InstrumentInfo]:
        """Search for instruments by name or symbol."""
        # Yahoo Finance doesn't have a direct search API through yfinance
        # We'll try to get info for the query as-is, assuming it's a symbol
        try:
            info = self.get_instrument_info(query.upper())
            return [info] if info else []
        except Exception:
            return []

    def get_instrument_by_isin(self, isin: str) -> Optional[InstrumentInfo]:
        """Get instrument information by ISIN."""
        # Yahoo Finance doesn't have direct ISIN lookup, but we can try
        # to search for it and see if any result matches the ISIN
        try:
            # Try to search for the ISIN as a query
            search_results = self.search_instruments(isin)
            for result in search_results:
                if result.isin and result.isin.upper() == isin.upper():
                    return result
            return None
        except Exception:
            return None

    def get_exchange_rate(
        self, from_currency: Currency, to_currency: Currency
    ) -> Optional[Decimal]:
        """Get current exchange rate between two currencies."""
        if from_currency == to_currency:
            return Decimal("1")

        try:
            # Yahoo Finance uses format like "EURUSD=X" for forex pairs
            if from_currency == Currency.USD:
                # For USD to other currencies, use inverse
                symbol = f"{to_currency.value}USD=X"
                ticker = self._get_ticker(symbol)
                rate = ticker.info.get("regularMarketPrice")
                if rate:
                    return Decimal("1") / Decimal(str(rate))
            else:
                symbol = f"{from_currency.value}{to_currency.value}=X"
                ticker = self._get_ticker(symbol)
                rate = ticker.info.get("regularMarketPrice")
                if rate:
                    return Decimal(str(rate))

            return None

        except Exception as e:
            print(f"Error getting exchange rate {from_currency} to {to_currency}: {e}")
            return None

    def supports_instrument_type(self, instrument_type: InstrumentType) -> bool:
        """Check if provider supports a specific instrument type."""
        supported_types = {
            InstrumentType.STOCK,
            InstrumentType.ETF,
            InstrumentType.MUTUAL_FUND,
            InstrumentType.CRYPTO,  # Limited crypto support
            InstrumentType.BOND,    # Bond ETFs and some individual bonds
        }
        return instrument_type in supported_types
