"""
Centralized market price storage with SQLite backend.

This module provides a single source of truth for all market prices,
replacing the duplicated price data in snapshots.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from .models import Currency


@dataclass
class PriceEntry:
    """A single price entry for storage."""

    symbol: str
    date: date
    price: Decimal
    currency: Currency
    source: Optional[str] = None


class MarketDataStore:
    """Centralized price storage - single source of truth for all market prices.

    Features:
    - SQLite backend for persistence
    - In-memory cache for fast access
    - Efficient bulk operations
    - Price matrix generation for analytics
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize the market data store.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "market_data.db"

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: symbol -> {date -> price}
        self._cache: Dict[str, Dict[date, Decimal]] = {}
        self._cache_loaded: Set[str] = set()

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def clear_cache(self) -> None:
        """Clear the in-memory cache to force reload from database.

        Call this when external processes may have updated the database,
        such as when the MCP server adds new market data.
        """
        self._cache.clear()
        self._cache_loaded.clear()
        logging.debug("MarketDataStore cache cleared")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Create market_prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_prices (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    price TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    source TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
            """)

            # Create index for date-based queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_prices_date
                ON market_prices(date)
            """)

            # Create index for symbol queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_prices_symbol
                ON market_prices(symbol)
            """)

            conn.commit()
            logging.debug("MarketDataStore database initialized")

        except Exception as e:
            logging.error(f"Error initializing market data database: {e}")
            raise
        finally:
            conn.close()

    def _load_symbol_cache(self, symbol: str) -> None:
        """Load all prices for a symbol into cache."""
        if symbol in self._cache_loaded:
            return

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT date, price FROM market_prices WHERE symbol = ?",
                (symbol,)
            )
            rows = cursor.fetchall()

            self._cache[symbol] = {}
            for row in rows:
                self._cache[symbol][date.fromisoformat(row["date"])] = Decimal(row["price"])

            self._cache_loaded.add(symbol)

        except Exception as e:
            logging.error(f"Error loading cache for {symbol}: {e}")
        finally:
            conn.close()

    def get_price(self, symbol: str, target_date: date) -> Optional[Decimal]:
        """Get price for a symbol on a specific date.

        Args:
            symbol: The trading symbol
            target_date: The date to get price for

        Returns:
            The price or None if not found
        """
        symbol = symbol.upper().strip()

        # Load into cache if not already
        self._load_symbol_cache(symbol)

        # Check cache
        if symbol in self._cache and target_date in self._cache[symbol]:
            return self._cache[symbol][target_date]

        return None

    def get_prices(
        self, symbol: str, start_date: date, end_date: date
    ) -> Dict[date, Decimal]:
        """Get prices for a symbol within a date range.

        Args:
            symbol: The trading symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Dict mapping date to price
        """
        symbol = symbol.upper().strip()

        # Load into cache if not already
        self._load_symbol_cache(symbol)

        result: Dict[date, Decimal] = {}
        if symbol in self._cache:
            for d, price in self._cache[symbol].items():
                if start_date <= d <= end_date:
                    result[d] = price

        return result

    def get_prices_with_currency(
        self, symbol: str, start_date: date, end_date: date
    ) -> Dict[date, PriceEntry]:
        """Get prices with currency for a symbol within a date range.

        Args:
            symbol: The trading symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Dict mapping date to PriceEntry (includes currency)
        """
        symbol = symbol.upper().strip()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT date, price, currency FROM market_prices WHERE symbol = ? AND date BETWEEN ? AND ?",
                (symbol, start_date.isoformat(), end_date.isoformat()),
            )
            rows = cursor.fetchall()
            return {
                date.fromisoformat(row["date"]): PriceEntry(
                    symbol=symbol,
                    date=date.fromisoformat(row["date"]),
                    price=Decimal(row["price"]),
                    currency=Currency(row["currency"]),
                )
                for row in rows
            }
        finally:
            conn.close()

    def get_price_with_fallback(
        self, symbol: str, target_date: date, fallback_days: int = 7
    ) -> Optional[Decimal]:
        """Get price for a date, falling back to most recent price if not found.

        Args:
            symbol: The trading symbol
            target_date: The date to get price for
            fallback_days: Maximum days to look back

        Returns:
            The price or None if not found within fallback window
        """
        symbol = symbol.upper().strip()
        self._load_symbol_cache(symbol)

        if symbol not in self._cache:
            return None

        # Try exact date first
        if target_date in self._cache[symbol]:
            return self._cache[symbol][target_date]

        # Look for most recent price before target_date
        prices = self._cache[symbol]
        min_date = target_date
        from datetime import timedelta
        min_date = target_date - timedelta(days=fallback_days)

        best_date: Optional[date] = None
        for d in prices.keys():
            if min_date <= d < target_date:
                if best_date is None or d > best_date:
                    best_date = d

        if best_date:
            return prices[best_date]

        return None

    def set_price(
        self,
        symbol: str,
        price_date: date,
        price: Decimal,
        currency: Currency,
        source: Optional[str] = None,
    ) -> bool:
        """Set price for a symbol on a specific date.

        Args:
            symbol: The trading symbol
            price_date: The date
            price: The price
            currency: The price currency
            source: Optional source identifier

        Returns:
            True if successful
        """
        symbol = symbol.upper().strip()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Upsert price
            cursor.execute("""
                INSERT OR REPLACE INTO market_prices
                (symbol, date, price, currency, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                price_date.isoformat(),
                str(price),
                currency.value,
                source,
                datetime.now().isoformat(),
            ))

            conn.commit()

            # Update cache
            if symbol not in self._cache:
                self._cache[symbol] = {}
            self._cache[symbol][price_date] = price
            self._cache_loaded.add(symbol)

            logging.debug(f"Stored price for {symbol} on {price_date}: {price}")
            return True

        except Exception as e:
            conn.rollback()
            logging.error(f"Error setting price for {symbol}: {e}")
            return False
        finally:
            conn.close()

    def set_prices_batch(self, prices: List[PriceEntry]) -> int:
        """Set multiple prices in a single transaction.

        Args:
            prices: List of price entries to store

        Returns:
            Number of prices stored
        """
        if not prices:
            return 0

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            for entry in prices:
                symbol = entry.symbol.upper().strip()
                cursor.execute("""
                    INSERT OR REPLACE INTO market_prices
                    (symbol, date, price, currency, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    entry.date.isoformat(),
                    str(entry.price),
                    entry.currency.value,
                    entry.source,
                    now,
                ))

                # Update cache
                if symbol not in self._cache:
                    self._cache[symbol] = {}
                self._cache[symbol][entry.date] = entry.price
                self._cache_loaded.add(symbol)

            conn.commit()
            logging.info(f"Batch stored {len(prices)} prices")
            return len(prices)

        except Exception as e:
            conn.rollback()
            logging.error(f"Error batch setting prices: {e}")
            return 0
        finally:
            conn.close()

    def get_price_matrix(
        self, symbols: List[str], start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Get prices for multiple symbols as a DataFrame.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date index and symbol columns, forward-filled
        """
        if not symbols:
            return pd.DataFrame()

        # Collect prices for all symbols
        data: Dict[str, Dict[date, float]] = {}
        for symbol in symbols:
            symbol = symbol.upper().strip()
            prices = self.get_prices(symbol, start_date, end_date)
            data[symbol] = {d: float(p) for d, p in prices.items()}

        if not data:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(data)

        # Create complete date range
        from datetime import timedelta
        all_dates = []
        current = start_date
        while current <= end_date:
            all_dates.append(current)
            current += timedelta(days=1)

        # Reindex to complete date range
        df.index = pd.to_datetime(df.index)
        date_index = pd.to_datetime(all_dates)
        df = df.reindex(date_index)

        # Forward-fill missing values
        df = df.ffill()

        return df

    def ensure_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        data_provider: Callable[[str, date, date], List[Tuple[date, Decimal]]],
        currency: Currency = Currency.USD,
    ) -> bool:
        """Ensure prices exist for a symbol in a date range, fetching if needed.

        Args:
            symbol: The trading symbol
            start_date: Start date
            end_date: End date
            data_provider: Callback to fetch missing prices
            currency: Currency for the prices

        Returns:
            True if prices are available
        """
        symbol = symbol.upper().strip()

        # Check what we have
        existing = self.get_prices(symbol, start_date, end_date)

        # Build list of missing dates
        from datetime import timedelta
        missing_dates: List[date] = []
        current = start_date
        while current <= end_date:
            if current not in existing:
                missing_dates.append(current)
            current += timedelta(days=1)

        if not missing_dates:
            return True  # All prices exist

        # Fetch missing prices using provider
        try:
            # Fetch for the entire missing range
            min_missing = min(missing_dates)
            max_missing = max(missing_dates)
            new_prices = data_provider(symbol, min_missing, max_missing)

            if new_prices:
                entries = [
                    PriceEntry(
                        symbol=symbol,
                        date=d,
                        price=p,
                        currency=currency,
                        source="data_provider",
                    )
                    for d, p in new_prices
                ]
                self.set_prices_batch(entries)
                return True

        except Exception as e:
            logging.error(f"Error ensuring prices for {symbol}: {e}")

        return False

    def get_symbols(self) -> List[str]:
        """Get all symbols in the store.

        Returns:
            List of symbol names
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM market_prices ORDER BY symbol")
            rows = cursor.fetchall()
            return [row["symbol"] for row in rows]
        except Exception as e:
            logging.error(f"Error getting symbols: {e}")
            return []
        finally:
            conn.close()

    def get_date_range(self, symbol: str) -> Optional[Tuple[date, date]]:
        """Get the date range of available prices for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            Tuple of (earliest_date, latest_date) or None if no data
        """
        symbol = symbol.upper().strip()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM market_prices
                WHERE symbol = ?
            """, (symbol,))

            row = cursor.fetchone()
            if not row or not row["min_date"]:
                return None

            return (
                date.fromisoformat(row["min_date"]),
                date.fromisoformat(row["max_date"]),
            )
        except Exception as e:
            logging.error(f"Error getting date range for {symbol}: {e}")
            return None
        finally:
            conn.close()

    def get_latest_price(self, symbol: str) -> Optional[Tuple[date, Decimal]]:
        """Get the most recent price for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            Tuple of (date, price) or None if no data
        """
        symbol = symbol.upper().strip()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, price FROM market_prices
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 1
            """, (symbol,))

            row = cursor.fetchone()
            if not row:
                return None

            return (
                date.fromisoformat(row["date"]),
                Decimal(row["price"]),
            )
        except Exception as e:
            logging.error(f"Error getting latest price for {symbol}: {e}")
            return None
        finally:
            conn.close()

    def delete_prices(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> int:
        """Delete prices with optional filters.

        Args:
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Number of prices deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            query = "DELETE FROM market_prices WHERE 1=1"
            params: List = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.upper().strip())
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.isoformat())

            cursor.execute(query, params)
            deleted = cursor.rowcount
            conn.commit()

            # Clear affected cache entries
            if symbol:
                symbol = symbol.upper().strip()
                if symbol in self._cache:
                    del self._cache[symbol]
                self._cache_loaded.discard(symbol)
            else:
                self._cache.clear()
                self._cache_loaded.clear()

            logging.info(f"Deleted {deleted} prices")
            return deleted

        except Exception as e:
            conn.rollback()
            logging.error(f"Error deleting prices: {e}")
            return 0
        finally:
            conn.close()

    def get_price_count(self, symbol: Optional[str] = None) -> int:
        """Get the number of stored prices.

        Args:
            symbol: Optional symbol to count for

        Returns:
            Number of price entries
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if symbol:
                cursor.execute(
                    "SELECT COUNT(*) as count FROM market_prices WHERE symbol = ?",
                    (symbol.upper().strip(),)
                )
            else:
                cursor.execute("SELECT COUNT(*) as count FROM market_prices")
            row = cursor.fetchone()
            return row["count"] if row else 0
        except Exception as e:
            logging.error(f"Error getting price count: {e}")
            return 0
        finally:
            conn.close()

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._cache_loaded.clear()
        logging.debug("MarketDataStore cache cleared")

    def migrate_from_snapshots(self, snapshot_store) -> int:
        """Migrate price data from existing snapshots.

        This extracts unique (symbol, date, price) combinations from snapshots
        and stores them in the MarketDataStore.

        Args:
            snapshot_store: SnapshotStore instance to migrate from

        Returns:
            Number of prices migrated
        """
        conn = snapshot_store._get_connection()
        try:
            cursor = conn.cursor()

            # Get all unique symbol/date/price combinations from snapshots
            cursor.execute("""
                SELECT DISTINCT
                    sp.symbol,
                    s.date,
                    sp.current_price,
                    sp.instrument_json
                FROM snapshot_positions sp
                JOIN snapshots s ON sp.snapshot_id = s.id
                WHERE sp.current_price IS NOT NULL
                ORDER BY sp.symbol, s.date
            """)
            rows = cursor.fetchall()

            if not rows:
                return 0

            # Convert to price entries
            entries: List[PriceEntry] = []
            for row in rows:
                try:
                    instrument_data = json.loads(row["instrument_json"])
                    currency_str = instrument_data.get("currency", "USD")
                    currency = Currency(currency_str)

                    entries.append(PriceEntry(
                        symbol=row["symbol"],
                        date=date.fromisoformat(row["date"]),
                        price=Decimal(row["current_price"]),
                        currency=currency,
                        source="snapshot_migration",
                    ))
                except Exception as e:
                    logging.warning(f"Error parsing snapshot row: {e}")
                    continue

            # Store all prices
            if entries:
                count = self.set_prices_batch(entries)
                logging.info(f"Migrated {count} prices from snapshots")
                return count

            return 0

        except Exception as e:
            logging.error(f"Error migrating from snapshots: {e}")
            return 0
        finally:
            conn.close()

    def interpolate_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        currency: Optional[Currency] = None,
    ) -> int:
        """Interpolate missing prices between two dates using linear interpolation.

        Finds the nearest available prices before start_date and after end_date,
        then fills in missing dates with linearly interpolated values.

        Args:
            symbol: The trading symbol
            start_date: Start of the date range to fill
            end_date: End of the date range to fill
            currency: Currency for the prices (auto-detected if not provided)

        Returns:
            Number of prices interpolated and stored
        """
        symbol = symbol.upper().strip()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Find the nearest price on or before start_date
            cursor.execute("""
                SELECT date, price, currency FROM market_prices
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC LIMIT 1
            """, (symbol, start_date.isoformat()))
            start_row = cursor.fetchone()

            # Find the nearest price on or after end_date
            cursor.execute("""
                SELECT date, price, currency FROM market_prices
                WHERE symbol = ? AND date >= ?
                ORDER BY date ASC LIMIT 1
            """, (symbol, end_date.isoformat()))
            end_row = cursor.fetchone()

            if not start_row or not end_row:
                logging.warning(
                    f"Cannot interpolate {symbol}: missing boundary prices "
                    f"(start: {start_row is not None}, end: {end_row is not None})"
                )
                return 0

            # Extract boundary data
            boundary_start_date = date.fromisoformat(start_row["date"])
            boundary_end_date = date.fromisoformat(end_row["date"])
            start_price = Decimal(start_row["price"])
            end_price = Decimal(end_row["price"])
            detected_currency = currency or Currency(start_row["currency"])

            # Calculate total days and daily change
            total_days = (boundary_end_date - boundary_start_date).days
            if total_days <= 0:
                logging.warning(f"Cannot interpolate {symbol}: invalid date range")
                return 0

            daily_change = (end_price - start_price) / Decimal(str(total_days))

            # Generate interpolated prices for missing dates
            entries: List[PriceEntry] = []
            from datetime import timedelta

            current = boundary_start_date + timedelta(days=1)
            while current < boundary_end_date:
                # Check if price already exists
                existing = self.get_price(symbol, current)
                if existing is None:
                    days_from_start = (current - boundary_start_date).days
                    interpolated_price = start_price + (daily_change * Decimal(str(days_from_start)))

                    entries.append(PriceEntry(
                        symbol=symbol,
                        date=current,
                        price=interpolated_price,
                        currency=detected_currency,
                        source="interpolated",
                    ))

                current += timedelta(days=1)

            # Store interpolated prices
            if entries:
                count = self.set_prices_batch(entries)
                logging.info(
                    f"Interpolated {count} prices for {symbol} "
                    f"from {boundary_start_date} to {boundary_end_date}"
                )
                return count

            return 0

        except Exception as e:
            logging.error(f"Error interpolating prices for {symbol}: {e}")
            return 0
        finally:
            conn.close()

    def interpolate_prices_batch(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, int]:
        """Interpolate missing prices for multiple symbols.

        Args:
            symbols: List of trading symbols
            start_date: Start of the date range to fill
            end_date: End of the date range to fill

        Returns:
            Dict mapping symbol to number of prices interpolated
        """
        results: Dict[str, int] = {}
        for symbol in symbols:
            count = self.interpolate_prices(symbol, start_date, end_date)
            results[symbol] = count
        return results
