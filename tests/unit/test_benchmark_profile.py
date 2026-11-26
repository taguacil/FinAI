"""Benchmark profile: computed on the portfolio's calendar dates."""

from datetime import date, timedelta
from unittest.mock import Mock

import pytest

from src.utils.metrics import FinancialMetricsCalculator


def test_profile_forward_fills_onto_calendar_dates():
    calc = FinancialMetricsCalculator(data_manager=Mock())
    # Fri 100, Mon 110, Tue 99 (weekend carries Friday's price forward)
    prices = {date(2026, 1, 2): 100.0, date(2026, 1, 5): 110.0, date(2026, 1, 6): 99.0}
    days = [date(2026, 1, 2) + timedelta(d) for d in range(5)]  # Fri..Tue

    p = calc._benchmark_profile(prices, days, risk_free_rate=0.0)

    assert p["total_return"] == pytest.approx(-0.01)
    assert p["max_drawdown"] == pytest.approx(0.10)
    # Same drawdown helper as the portfolio, on the forward-filled calendar series.
    filled = [100.0, 100.0, 100.0, 110.0, 99.0]
    assert p["max_drawdown_duration"] == calc._calculate_max_drawdown_from_values(filled)[1]
    assert p["best_day"] == pytest.approx(0.10)
    assert p["worst_day"] == pytest.approx(-0.10)
    assert (p["positive_days"], p["moving_days"]) == (1, 2)  # weekend zeros excluded


def test_profile_starts_at_first_date_with_a_price():
    calc = FinancialMetricsCalculator(data_manager=Mock())
    prices = {date(2026, 1, 5): 100.0, date(2026, 1, 6): 102.0}
    days = [date(2026, 1, 3) + timedelta(d) for d in range(4)]  # Sat..Tue

    p = calc._benchmark_profile(prices, days, risk_free_rate=0.0)

    assert p["total_return"] == pytest.approx(0.02)


def test_profile_none_without_prices():
    calc = FinancialMetricsCalculator(data_manager=Mock())
    assert calc._benchmark_profile({}, [date(2026, 1, 2)], 0.0) is None
