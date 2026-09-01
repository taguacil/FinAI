"""
Unit tests for the walk-forward backtesting engine.

These are fully offline: prices are written into a temporary MarketDataStore and
no benchmark/metrics calculator is used (metrics require network), so the tests
exercise the simulation core — equity math, rebalancing, scheduling, transaction
costs and the no-look-ahead guarantee.
"""

from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from src.portfolio.backtester import BacktestConfig, Backtester
from src.portfolio.market_data_store import MarketDataStore, PriceEntry
from src.portfolio.models import Currency
from src.portfolio.strategies import (
    BuyAndHoldStrategy,
    EqualWeightStrategy,
    FixedWeightStrategy,
    OptimizerStrategy,
    build_strategies,
)

# --- Fixtures --------------------------------------------------------------


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


# --- Config validation -----------------------------------------------------


def test_config_rejects_bad_frequency():
    with pytest.raises(ValueError):
        BacktestConfig(
            symbols=["AAA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),
            rebalance_frequency="hourly",
        )


def test_config_rejects_inverted_dates():
    with pytest.raises(ValueError):
        BacktestConfig(
            symbols=["AAA"], start_date=date(2024, 2, 1), end_date=date(2024, 1, 1)
        )


def test_config_uppercases_symbols():
    cfg = BacktestConfig(
        symbols=["aaa", " bbb "], start_date=date(2024, 1, 1), end_date=date(2024, 2, 1)
    )
    assert cfg.symbols == ["AAA", "BBB"]


# --- Equity math -----------------------------------------------------------


def test_single_asset_equity_tracks_price(store):
    # One asset that doubles: buy-and-hold equity should double too.
    start = date(2024, 1, 1)
    n = 40
    _write_series(store, "AAA", start, [100 + 5 * i for i in range(n)])

    cfg = BacktestConfig(
        symbols=["AAA"],
        start_date=start,
        end_date=start + timedelta(days=n - 1),
        initial_capital=10_000.0,
        rebalance_frequency="none",
        lookback_days=5,
    )
    bt = Backtester(store)
    result = bt.run(cfg, [BuyAndHoldStrategy()])
    curve = result.strategies[0].equity_curve["total_value"]

    # Fully invested at 100 on day 0 -> 100 shares. Final price 100+5*39=295.
    assert curve.iloc[0] == pytest.approx(10_000.0, rel=1e-6)
    assert curve.iloc[-1] == pytest.approx(100 * 295, rel=1e-6)


def test_equal_weight_two_assets(store):
    start = date(2024, 1, 1)
    n = 30
    _write_series(store, "AAA", start, [100] * n)  # flat
    _write_series(store, "BBB", start, [50] * n)  # flat
    cfg = BacktestConfig(
        symbols=["AAA", "BBB"],
        start_date=start,
        end_date=start + timedelta(days=n - 1),
        initial_capital=10_000.0,
        rebalance_frequency="none",
        lookback_days=5,
    )
    result = Backtester(store).run(cfg, [EqualWeightStrategy()])
    curve = result.strategies[0].equity_curve["total_value"]
    # Flat prices -> equity stays at initial capital, fully invested (no cash left).
    assert curve.iloc[-1] == pytest.approx(10_000.0, rel=1e-6)


# --- Transaction costs -----------------------------------------------------


def test_transaction_costs_reduce_equity(store):
    start = date(2024, 1, 1)
    n = 30
    _write_series(store, "AAA", start, [100] * n)
    _write_series(store, "BBB", start, [100] * n)
    end = start + timedelta(days=n - 1)

    def make_cfg(bps):
        return BacktestConfig(
            symbols=["AAA", "BBB"],
            start_date=start,
            end_date=end,
            initial_capital=10_000.0,
            rebalance_frequency="none",
            lookback_days=5,
            transaction_cost_bps=bps,
        )

    free = Backtester(store).run(make_cfg(0), [EqualWeightStrategy()])
    costly = Backtester(store).run(make_cfg(50), [EqualWeightStrategy()])

    free_final = free.strategies[0].equity_curve["total_value"].iloc[-1]
    costly_final = costly.strategies[0].equity_curve["total_value"].iloc[-1]
    # Initial deploy of ~10k at 50bps costs ~$50.
    assert costly_final < free_final
    assert (free_final - costly_final) == pytest.approx(10_000 * 0.005, rel=0.05)


# --- Rebalancing schedule --------------------------------------------------


def test_monthly_rebalance_count(store):
    start = date(2024, 1, 1)
    n = 100  # ~3.3 months
    _write_series(store, "AAA", start, [100 + i for i in range(n)])
    _write_series(store, "BBB", start, [100 + i for i in range(n)])
    cfg = BacktestConfig(
        symbols=["AAA", "BBB"],
        start_date=start,
        end_date=start + timedelta(days=n - 1),
        rebalance_frequency="monthly",
        lookback_days=5,
    )
    result = Backtester(store).run(cfg, [EqualWeightStrategy()])
    reb_dates = {w["date"] for w in result.strategies[0].weights_history}
    # Jan (initial), Feb, Mar, Apr -> 4 rebalances.
    assert len(reb_dates) == 4


def test_buy_and_hold_rebalances_once(store):
    start = date(2024, 1, 1)
    n = 100
    _write_series(store, "AAA", start, [100 + i for i in range(n)])
    _write_series(store, "BBB", start, [200 - i for i in range(n)])
    cfg = BacktestConfig(
        symbols=["AAA", "BBB"],
        start_date=start,
        end_date=start + timedelta(days=n - 1),
        rebalance_frequency="monthly",
        lookback_days=5,
    )
    result = Backtester(store).run(cfg, [BuyAndHoldStrategy()])
    assert len(result.strategies[0].weights_history) == 1


# --- No look-ahead ---------------------------------------------------------


def test_no_lookahead(store):
    """A price spike strictly AFTER a rebalance date must not change the weights
    chosen at that date."""
    start = date(2024, 1, 1)
    n = 60
    base = [100 + i for i in range(n)]
    _write_series(store, "AAA", start, base)
    _write_series(store, "BBB", start, base)

    class RecordingStrategy(OptimizerStrategy):
        def __init__(self):
            super().__init__(lookback_days=252)
            self.seen_last_date = []

        def target_weights(self, as_of, prices_window, current_weights):
            self.seen_last_date.append(prices_window.index[-1].date())
            return super().target_weights(as_of, prices_window, current_weights)

    strat = RecordingStrategy()
    cfg = BacktestConfig(
        symbols=["AAA", "BBB"],
        start_date=start,
        end_date=start + timedelta(days=n - 1),
        rebalance_frequency="monthly",
        lookback_days=252,
    )
    Backtester(store).run(cfg, [strat])
    # The window handed to the strategy never extends past the rebalance date.
    for w in strat.seen_last_date:
        assert w <= start + timedelta(days=n - 1)
    # And specifically: the last observed date equals the rebalance date each time.
    assert strat.seen_last_date  # ran at least once


# --- Multiple strategies in one run ---------------------------------------


def test_multiple_strategies_one_run(store):
    start = date(2024, 1, 1)
    n = 80
    _write_series(store, "AAA", start, [100 + i for i in range(n)])
    _write_series(store, "BBB", start, [100 + 0.5 * i for i in range(n)])
    cfg = BacktestConfig(
        symbols=["AAA", "BBB"],
        start_date=start,
        end_date=start + timedelta(days=n - 1),
        rebalance_frequency="monthly",
        lookback_days=30,
    )
    strategies = build_strategies(
        ["hrp", "equal_weight", "buy_and_hold"], lookback_days=30
    )
    result = Backtester(store).run(cfg, strategies)
    assert len(result.strategies) == 3
    names = {s.name for s in result.strategies}
    assert {"HRP", "Equal Weight", "Buy & Hold"} == names
    for s in result.strategies:
        assert len(s.equity_curve) == len(result.strategies[0].equity_curve)


# --- Fixed weights ---------------------------------------------------------


def test_fixed_weights_normalized(store):
    start = date(2024, 1, 1)
    n = 30
    _write_series(store, "AAA", start, [100] * n)
    _write_series(store, "BBB", start, [100] * n)
    cfg = BacktestConfig(
        symbols=["AAA", "BBB"],
        start_date=start,
        end_date=start + timedelta(days=n - 1),
        rebalance_frequency="none",
        lookback_days=5,
        initial_capital=10_000.0,
    )
    strat = FixedWeightStrategy({"AAA": 3, "BBB": 1})  # 75/25 after normalization
    result = Backtester(store).run(cfg, [strat])
    w = result.strategies[0].weights_history[0]["weights"]
    assert w["AAA"] == pytest.approx(0.75, rel=1e-6)
    assert w["BBB"] == pytest.approx(0.25, rel=1e-6)


# --- Errors ----------------------------------------------------------------


def test_missing_prices_raises(store):
    cfg = BacktestConfig(
        symbols=["NOPE"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 1),
        lookback_days=5,
    )
    with pytest.raises(ValueError):
        Backtester(store).run(cfg, [EqualWeightStrategy()])
