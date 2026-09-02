"""Unit tests for MarketDataStore price lookups.

Fully offline: prices are written into a temporary MarketDataStore and read
back, with no network access.
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from src.portfolio.market_data_store import MarketDataStore, PriceEntry
from src.portfolio.models import Currency


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(data_dir=str(tmp_path))


def _write_series(store, symbol, start, prices, currency=Currency.USD):
    """Write a daily price series starting at ``start`` (calendar days)."""
    entries = [
        PriceEntry(
            symbol=symbol,
            date=start + timedelta(days=i),
            price=Decimal(str(p)),
            currency=currency,
        )
        for i, p in enumerate(prices)
    ]
    store.set_prices_batch(entries)


# --- get_price_with_fallback -----------------------------------------------


def test_fallback_returns_exact_price(store):
    _write_series(store, "AAA", date(2026, 1, 1), [10, 11, 12])
    assert store.get_price_with_fallback("AAA", date(2026, 1, 2)) == Decimal("11")


def test_fallback_looks_back_within_window(store):
    _write_series(store, "AAA", date(2026, 1, 1), [10])
    # 5 days later, within the default 7-day window
    assert store.get_price_with_fallback("AAA", date(2026, 1, 6)) == Decimal("10")


def test_fallback_returns_none_beyond_window(store):
    _write_series(store, "AAA", date(2026, 1, 1), [10])
    # 30 days later, well beyond the 7-day window
    assert store.get_price_with_fallback("AAA", date(2026, 1, 31)) is None


# --- get_last_price_on_or_before (carry-forward) ---------------------------


def test_last_price_carries_forward_beyond_fallback_window(store):
    """A held instrument whose quotes end weeks earlier still resolves."""
    _write_series(store, "BOND", date(2026, 1, 1), [1.00, 1.01, 1.02])
    # 60 days after the last quote — fallback gives up, carry-forward does not.
    target = date(2026, 3, 3)
    assert store.get_price_with_fallback("BOND", target) is None
    assert store.get_last_price_on_or_before("BOND", target) == Decimal("1.02")


def test_last_price_returns_none_before_first_quote(store):
    _write_series(store, "BOND", date(2026, 1, 10), [1.00])
    assert store.get_last_price_on_or_before("BOND", date(2026, 1, 1)) is None


def test_last_price_unknown_symbol(store):
    assert store.get_last_price_on_or_before("NOPE", date(2026, 1, 1)) is None
