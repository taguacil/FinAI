"""Unit tests for the analytics benchmark-curve anchoring.

The benchmark cumulative-return curve must share a 0% origin with the portfolio
curve. When the analysis window starts before a position was opened (which is
normal for a single-class view, since each class is bought on a different date),
the benchmark must be rebased to the day the portfolio first has value -- not to
the window start -- otherwise it is credited with gains earned while the
portfolio held nothing, and appears to change whenever you filter by class.
"""

import pandas as pd
import pytest

from src.web.services import _rebased_benchmark_curve


def _index(*days):
    return pd.to_datetime(list(days))


def test_benchmark_anchors_at_portfolio_start_not_window_start():
    # 5-day window; portfolio holds nothing until day 3 (index 2), so its first
    # return is on index 3 -> ret_idx starts at 3 and anchors the benchmark at
    # index 2 (the day before). SPY rose 10% before that (100 -> 110) and 10%
    # after (110 -> 121); only the post-holding 10% must show.
    idx = _index("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05")
    prices = {
        "2025-01-01": 100.0,
        "2025-01-02": 105.0,
        "2025-01-03": 110.0,  # portfolio opens here -> benchmark base
        "2025-01-04": 115.5,
        "2025-01-05": 121.0,
    }
    ret_idx = [3, 4]  # portfolio had value on index 2 onward
    curve = _rebased_benchmark_curve(prices, idx, ret_idx, n=len(ret_idx))
    assert curve is not None
    # First plotted point is the first return day (index 3): 115.5/110 - 1 = 5%.
    assert curve[0] == pytest.approx(5.0)
    # Last point (index 4): 121/110 - 1 = 10%, NOT 21% (which window-start anchoring gives).
    assert curve[-1] == pytest.approx(10.0)


def test_full_window_view_anchors_at_window_start():
    # Portfolio has value from day 1, so ret_idx starts at 1 and the benchmark
    # anchors at the window start -- the curve begins at ~0%.
    idx = _index("2025-01-01", "2025-01-02", "2025-01-03")
    prices = {"2025-01-01": 100.0, "2025-01-02": 110.0, "2025-01-03": 120.0}
    ret_idx = [1, 2]
    curve = _rebased_benchmark_curve(prices, idx, ret_idx, n=2)
    assert curve[0] == pytest.approx(10.0)
    assert curve[-1] == pytest.approx(20.0)


def test_no_flat_tail_past_last_observation():
    # Window extends a day past the last real benchmark price; that trailing day
    # must be None, not a fabricated flat value.
    idx = _index("2025-01-01", "2025-01-02", "2025-01-03")
    prices = {"2025-01-01": 100.0, "2025-01-02": 110.0}  # nothing for 01-03
    ret_idx = [1, 2]
    curve = _rebased_benchmark_curve(prices, idx, ret_idx, n=2)
    assert curve[0] == pytest.approx(10.0)
    assert curve[-1] is None


def test_returns_none_without_prices_or_returns():
    idx = _index("2025-01-01", "2025-01-02")
    assert _rebased_benchmark_curve(None, idx, [1], n=1) is None
    assert _rebased_benchmark_curve({}, idx, [1], n=1) is None
    assert _rebased_benchmark_curve({"2025-01-01": 100.0}, idx, [], n=0) is None
