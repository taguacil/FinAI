"""Tests for FXRateCache lookups (exact date, nearest-date fallback, reloads)."""

import os
from datetime import date
from decimal import Decimal

import pytest

from src.data_providers.fx_cache import FXRateCache
from src.portfolio.models import Currency


def _write(path, rows):
    path.write_text("date,rate\n" + "".join(f"{d},{r}\n" for d, r in rows))


@pytest.fixture
def cache(tmp_path):
    # Fri 2026-01-02 and Mon 2026-01-05 bracket a weekend; 2026-01-20 is isolated.
    _write(tmp_path / "USD_EUR.csv", [
        ("2026-01-02", "0.90"),
        ("2026-01-05", "0.92"),
        ("2026-01-20", "0.95"),
    ])
    return FXRateCache(cache_dir=str(tmp_path))


def test_exact_date(cache):
    assert cache.get_rate(Currency.USD, Currency.EUR, date(2026, 1, 5)) == Decimal("0.92")


def test_nearest_date_within_window(cache):
    # Sun 01-04 is 2 days after 01-02 and 1 day before 01-05.
    assert cache.get_rate(Currency.USD, Currency.EUR, date(2026, 1, 4)) == Decimal("0.92")


def test_tie_resolves_to_earlier_date(tmp_path):
    _write(tmp_path / "USD_EUR.csv", [("2026-01-02", "0.90"), ("2026-01-06", "0.94")])
    c = FXRateCache(cache_dir=str(tmp_path))
    assert c.get_rate(Currency.USD, Currency.EUR, date(2026, 1, 4)) == Decimal("0.90")


def test_outside_window_returns_none(cache):
    # 01-12 is 7 days from 01-05 (in window); 01-13 is 7 from 01-20 (in window);
    # pick a date more than 7 days from every row.
    assert cache.get_rate(Currency.USD, Currency.EUR, date(2026, 3, 1)) is None
    assert cache.get_rate(Currency.USD, Currency.EUR, date(2025, 12, 1)) is None


def test_window_edge_is_inclusive(cache):
    assert cache.get_rate(Currency.USD, Currency.EUR, date(2026, 1, 27)) == Decimal("0.95")
    assert cache.get_rate(Currency.USD, Currency.EUR, date(2026, 1, 28)) is None


def test_inverse_pair(cache):
    rate = cache.get_rate(Currency.EUR, Currency.USD, date(2026, 1, 5))
    assert rate == Decimal("1") / Decimal("0.92")


def test_same_currency(cache):
    assert cache.get_rate(Currency.USD, Currency.USD, date(2026, 1, 5)) == Decimal("1.0")


def test_current_rate_is_newest(cache):
    assert cache.get_current_rate(Currency.USD, Currency.EUR) == Decimal("0.95")


def test_store_rate_visible_to_lookups(cache):
    assert cache.store_rate(Currency.USD, Currency.EUR, date(2026, 2, 10), Decimal("0.97"))
    assert cache.get_rate(Currency.USD, Currency.EUR, date(2026, 2, 10)) == Decimal("0.97")
    assert cache.get_current_rate(Currency.USD, Currency.EUR) == Decimal("0.97")


def test_external_file_change_is_picked_up(tmp_path, cache):
    path = tmp_path / "USD_EUR.csv"
    assert cache.get_current_rate(Currency.USD, Currency.EUR) == Decimal("0.95")
    # Another process (e.g. the MCP server) appends a newer rate.
    _write(path, [("2026-01-20", "0.95"), ("2026-02-02", "0.99")])
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert cache.get_current_rate(Currency.USD, Currency.EUR) == Decimal("0.99")


def test_malformed_rows_are_skipped(tmp_path):
    (tmp_path / "USD_EUR.csv").write_text(
        "date,rate\n2026-01-02,0.90\nnot-a-date,1\n2026-01-05,oops\n2026-01-06,0.93\n"
    )
    c = FXRateCache(cache_dir=str(tmp_path))
    assert c.get_rate(Currency.USD, Currency.EUR, date(2026, 1, 6)) == Decimal("0.93")
    assert c.get_rate(Currency.USD, Currency.EUR, date(2026, 1, 2)) == Decimal("0.90")


def test_memory_cache_stays_bounded(cache):
    cache.max_memory_cache_size = 3
    for day in range(1, 10):
        cache.get_rate(Currency.USD, Currency.EUR, date(2026, 1, day))
    assert len(cache._memory_cache) == 3
    assert set(cache._memory_cache) == set(cache._memory_cache_dates)
