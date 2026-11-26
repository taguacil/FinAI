"""FX handling in PortfolioOptimizer.

Regression tests for the Optimize bug where multi-currency positions were valued
as if every price were already in the base currency. The root cause was
``_get_fx_rate`` silently returning 1.0 when the cache lookup failed (it omitted
the required ``rate_date`` argument), which let a JPY position dominate the
portfolio weights and inflated the trade sizes.
"""

import tempfile
from datetime import date
from decimal import Decimal

import pytest

from src.portfolio.market_data_store import MarketDataStore
from src.portfolio.models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    Position,
)
from src.portfolio.optimizer import PortfolioOptimizer


class _FakeFXCache:
    """Minimal stand-in for FXRateCache.get_rate(from, to, rate_date)."""

    def __init__(self, rates):
        # rates keyed by (from, to) -> Decimal
        self._rates = rates
        self.calls = []

    def get_rate(self, from_currency, to_currency, rate_date):
        self.calls.append((from_currency, to_currency, rate_date))
        return self._rates.get((from_currency, to_currency))


class _FakeDataManager:
    """Looks like DataProviderManager to the optimizer (no ``data_manager`` attr)."""

    def __init__(self, rates):
        self.fx_cache = _FakeFXCache(rates)

    def get_exchange_rate(self, from_currency, to_currency, force_refresh=False):
        return self.fx_cache._rates.get((from_currency, to_currency))


def _instrument(symbol, currency, itype=InstrumentType.ETF):
    return FinancialInstrument(
        symbol=symbol,
        name=f"{symbol} test instrument",
        instrument_type=itype,
        currency=currency,
    )


def _position(symbol, currency, quantity, price):
    return Position(
        instrument=_instrument(symbol, currency),
        quantity=Decimal(str(quantity)),
        average_cost=Decimal(str(price)),
        current_price=Decimal(str(price)),
    )


def _optimizer(rates):
    tmp = tempfile.TemporaryDirectory()
    store = MarketDataStore(data_dir=tmp.name)
    opt = PortfolioOptimizer(
        _FakeDataManager(rates),
        base_currency=Currency.USD,
        market_data_store=store,
    )
    # keep the temp dir alive for the optimizer's lifetime
    opt._tmp = tmp
    return opt


def test_get_fx_rate_uses_cached_rate_not_one():
    """A cross-currency rate must come from the cache, never a 1.0 fallback."""
    opt = _optimizer({(Currency.JPY, Currency.USD): Decimal("0.0065")})
    rate = opt._get_fx_rate(Currency.JPY, Currency.USD)
    assert rate == Decimal("0.0065")
    # The cache was queried with a concrete rate_date (the previous bug omitted it).
    assert opt.data_provider.fx_cache.calls
    assert opt.data_provider.fx_cache.calls[0][2] == date.today()


def test_get_fx_rate_same_currency_is_one():
    opt = _optimizer({})
    assert opt._get_fx_rate(Currency.USD, Currency.USD) == Decimal("1")


def test_get_fx_rate_raises_when_missing():
    """A missing rate must raise, not silently return 1.0 and corrupt weights."""
    opt = _optimizer({})
    with pytest.raises(ValueError):
        opt._get_fx_rate(Currency.JPY, Currency.USD)


def test_position_value_converts_to_base():
    """A JPY position is valued in USD, not 1:1 with the base currency."""
    opt = _optimizer({(Currency.JPY, Currency.USD): Decimal("0.0065")})
    pos = _position("CNKY", Currency.JPY, quantity=100, price=67890)
    value = opt._get_position_value_in_base(pos)
    # 100 * 67890 JPY * 0.0065 = 44128.5 USD, not 6,789,000.
    assert value == Decimal("100") * Decimal("67890") * Decimal("0.0065")
    assert value < Decimal("50000")


def test_foreign_position_does_not_dominate_weights():
    """With FX applied, a small JPY holding no longer swamps a USD holding."""
    opt = _optimizer({(Currency.JPY, Currency.USD): Decimal("0.0065")})
    positions = {
        "SPY": _position("SPY", Currency.USD, quantity=1000, price=500),  # 500k USD
        "CNKY": _position("CNKY", Currency.JPY, quantity=100, price=67890),  # ~44k USD
    }
    weights, _ = opt._calculate_current_weights_with_cash(positions, Decimal("0"))
    # Without the fix CNKY would read as ~6.79M USD and dominate (>90%).
    assert weights["CNKY"] < 0.15
    assert weights["SPY"] > 0.85
