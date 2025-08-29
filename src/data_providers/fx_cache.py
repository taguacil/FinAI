"""
Foreign Exchange (FX) rate caching system with CSV storage.
Provides persistent storage and management of historical exchange rates.
"""

import csv
import logging
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from ..portfolio.models import Currency


class FXRateCache:
    """Persistent cache for foreign exchange rates using CSV storage."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the FX rate cache.

        Args:
            cache_dir: Directory to store cache files. Defaults to data/fx_cache/
        """
        self.cache_dir = Path(cache_dir or "data/fx_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for frequently accessed rates
        self._memory_cache: Dict[str, Decimal] = {}
        self._memory_cache_dates: Dict[str, date] = {}

        # Track which currency pairs we have data for
        self._available_pairs: Set[str] = set()

        # Cache settings
        self.max_memory_cache_size = 1000
        self.cache_freshness_days = 7  # Consider rates fresh for 7 days

        self._load_available_pairs()

    def _get_cache_filename(self, from_currency: Currency, to_currency: Currency) -> Path:
        """Get the cache filename for a currency pair."""
        pair = f"{from_currency.value}_{to_currency.value}"
        return self.cache_dir / f"{pair}.csv"

    def _get_cache_key(self, from_currency: Currency, to_currency: Currency, rate_date: date) -> str:
        """Get cache key for in-memory storage."""
        return f"{from_currency.value}_{to_currency.value}_{rate_date.isoformat()}"

    def _load_available_pairs(self):
        """Load list of available currency pairs from cache directory."""
        self._available_pairs.clear()

        for file_path in self.cache_dir.glob("*.csv"):
            # Extract currency pair from filename (e.g., "USD_EUR.csv" -> "USD_EUR")
            pair = file_path.stem
            if "_" in pair and len(pair.split("_")) == 2:
                self._available_pairs.add(pair)

        logging.debug(f"Loaded {len(self._available_pairs)} currency pairs from cache")

    def get_rate(self, from_currency: Currency, to_currency: Currency, rate_date: date) -> Optional[Decimal]:
        """Get exchange rate for a specific date.

        Args:
            from_currency: Currency to convert from
            to_currency: Currency to convert to
            rate_date: Date for the exchange rate

        Returns:
            Exchange rate as Decimal, or None if not found
        """
        if from_currency == to_currency:
            return Decimal("1.0")

        # Check in-memory cache first
        cache_key = self._get_cache_key(from_currency, to_currency, rate_date)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Try direct pair (FROM -> TO)
        rate = self._load_rate_from_csv(from_currency, to_currency, rate_date)
        if rate is not None:
            self._store_in_memory_cache(cache_key, rate, rate_date)
            return rate

        # Try inverse pair (TO -> FROM) and invert the rate
        inverse_rate = self._load_rate_from_csv(to_currency, from_currency, rate_date)
        if inverse_rate is not None and inverse_rate != 0:
            rate = Decimal("1") / inverse_rate
            self._store_in_memory_cache(cache_key, rate, rate_date)
            return rate

        return None

    def store_rate(self, from_currency: Currency, to_currency: Currency, rate_date: date, rate: Decimal):
        """Store an exchange rate in the cache.

        Args:
            from_currency: Currency to convert from
            to_currency: Currency to convert to
            rate_date: Date for the exchange rate
            rate: Exchange rate value
        """
        if from_currency == to_currency:
            return  # Don't store 1:1 rates

        # Store in CSV
        self._store_rate_in_csv(from_currency, to_currency, rate_date, rate)

        # Store in memory cache
        cache_key = self._get_cache_key(from_currency, to_currency, rate_date)
        self._store_in_memory_cache(cache_key, rate, rate_date)

        # Update available pairs
        pair = f"{from_currency.value}_{to_currency.value}"
        self._available_pairs.add(pair)

    def _load_rate_from_csv(self, from_currency: Currency, to_currency: Currency, rate_date: date) -> Optional[Decimal]:
        """Load exchange rate from CSV file."""
        cache_file = self._get_cache_filename(from_currency, to_currency)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)

                # Look for exact date first
                for row in reader:
                    if row['date'] == rate_date.isoformat():
                        return Decimal(row['rate'])

                # If exact date not found, look for closest date within reasonable range
                csvfile.seek(0)
                next(reader)  # Skip header

                closest_rate = None
                closest_days_diff = float('inf')

                for row in reader:
                    row_date = datetime.strptime(row['date'], '%Y-%m-%d').date()
                    days_diff = abs((row_date - rate_date).days)

                    # Only consider rates within a reasonable timeframe (e.g., 7 days)
                    if days_diff <= 7 and days_diff < closest_days_diff:
                        closest_days_diff = days_diff
                        closest_rate = Decimal(row['rate'])

                return closest_rate

        except Exception as e:
            logging.warning(f"Error reading FX cache file {cache_file}: {e}")
            return None

    def _store_rate_in_csv(self, from_currency: Currency, to_currency: Currency, rate_date: date, rate: Decimal):
        """Store exchange rate in CSV file."""
        cache_file = self._get_cache_filename(from_currency, to_currency)

        # Read existing data
        existing_rates = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        existing_rates[row['date']] = row['rate']
            except Exception as e:
                logging.warning(f"Error reading existing FX cache file {cache_file}: {e}")

        # Add/update the new rate
        existing_rates[rate_date.isoformat()] = str(rate)

        # Write back to file (sorted by date)
        try:
            with open(cache_file, 'w', newline='') as csvfile:
                fieldnames = ['date', 'rate']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for date_str in sorted(existing_rates.keys()):
                    writer.writerow({
                        'date': date_str,
                        'rate': existing_rates[date_str]
                    })

        except Exception as e:
            logging.error(f"Error writing FX cache file {cache_file}: {e}")

    def _store_in_memory_cache(self, cache_key: str, rate: Decimal, rate_date: date):
        """Store rate in in-memory cache with size management."""
        # Remove oldest entries if cache is full
        if len(self._memory_cache) >= self.max_memory_cache_size:
            # Find and remove the oldest entry
            oldest_key = min(self._memory_cache_dates.keys(),
                           key=lambda k: self._memory_cache_dates[k])
            del self._memory_cache[oldest_key]
            del self._memory_cache_dates[oldest_key]

        self._memory_cache[cache_key] = rate
        self._memory_cache_dates[cache_key] = rate_date

    def get_current_rate(self, from_currency: Currency, to_currency: Currency) -> Optional[Decimal]:
        """Get the most recent cached rate for a currency pair.

        Returns the newest rate available in the cache, useful for current/live rates.
        """
        if from_currency == to_currency:
            return Decimal("1.0")

        cache_file = self._get_cache_filename(from_currency, to_currency)

        if not cache_file.exists():
            # Try inverse pair
            inverse_file = self._get_cache_filename(to_currency, from_currency)
            if inverse_file.exists():
                inverse_rate = self._get_newest_rate_from_file(inverse_file)
                if inverse_rate is not None and inverse_rate != 0:
                    return Decimal("1") / inverse_rate
            return None

        return self._get_newest_rate_from_file(cache_file)

    def _get_newest_rate_from_file(self, cache_file: Path) -> Optional[Decimal]:
        """Get the newest rate from a CSV file."""
        try:
            with open(cache_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)

                newest_rate = None
                newest_date = None

                for row in reader:
                    row_date = datetime.strptime(row['date'], '%Y-%m-%d').date()
                    if newest_date is None or row_date > newest_date:
                        newest_date = row_date
                        newest_rate = Decimal(row['rate'])

                return newest_rate

        except Exception as e:
            logging.warning(f"Error reading newest rate from {cache_file}: {e}")
            return None

    def is_rate_fresh(self, from_currency: Currency, to_currency: Currency, rate_date: date) -> bool:
        """Check if we have a fresh rate for the given date."""
        cached_rate = self.get_rate(from_currency, to_currency, rate_date)
        if cached_rate is None:
            return False

        # Check if the rate is within our freshness window
        days_old = (date.today() - rate_date).days
        return days_old <= self.cache_freshness_days

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the cache."""
        total_pairs = len(self._available_pairs)
        memory_cached = len(self._memory_cache)

        # Count total CSV records
        total_records = 0
        for pair in self._available_pairs:
            currencies = pair.split('_')
            if len(currencies) == 2:
                try:
                    from_curr = Currency(currencies[0])
                    to_curr = Currency(currencies[1])
                    cache_file = self._get_cache_filename(from_curr, to_curr)

                    if cache_file.exists():
                        with open(cache_file, 'r', newline='') as csvfile:
                            total_records += sum(1 for _ in csv.DictReader(csvfile))
                except:
                    continue

        return {
            'currency_pairs': total_pairs,
            'memory_cached_rates': memory_cached,
            'total_csv_records': total_records,
            'cache_directory_size_mb': self._get_directory_size_mb()
        }

    def _get_directory_size_mb(self) -> float:
        """Get the size of the cache directory in MB."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)

    def cleanup_old_rates(self, days_to_keep: int = 365):
        """Remove rates older than specified days from all cache files."""
        cutoff_date = date.today() - timedelta(days=days_to_keep)

        for cache_file in self.cache_dir.glob("*.csv"):
            try:
                # Read current data
                rates_to_keep = {}
                with open(cache_file, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        row_date = datetime.strptime(row['date'], '%Y-%m-%d').date()
                        if row_date >= cutoff_date:
                            rates_to_keep[row['date']] = row['rate']

                # Write back only recent data
                if rates_to_keep:
                    with open(cache_file, 'w', newline='') as csvfile:
                        fieldnames = ['date', 'rate']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for date_str in sorted(rates_to_keep.keys()):
                            writer.writerow({
                                'date': date_str,
                                'rate': rates_to_keep[date_str]
                            })
                else:
                    # Remove empty files
                    cache_file.unlink()

            except Exception as e:
                logging.warning(f"Error cleaning up cache file {cache_file}: {e}")

        # Refresh available pairs after cleanup
        self._load_available_pairs()

    def clear_cache(self):
        """Clear all cached data (both memory and files)."""
        # Clear memory cache
        self._memory_cache.clear()
        self._memory_cache_dates.clear()

        # Remove all CSV files
        for cache_file in self.cache_dir.glob("*.csv"):
            try:
                cache_file.unlink()
            except Exception as e:
                logging.warning(f"Error removing cache file {cache_file}: {e}")

        self._available_pairs.clear()
        logging.info("FX rate cache cleared")
