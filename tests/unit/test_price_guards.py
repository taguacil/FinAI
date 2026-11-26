"""Guards in PortfolioManager._fetch_and_store_prices.

Covers the two failure modes that silently corrupted good price data:
1. A wrong-listing/wrong-scale fetch (e.g. a 10x jump) overwriting good data.
2. A foreign listing quoting a different currency being stored as-is.
"""

import tempfile
from datetime import date
from decimal import Decimal
from unittest.mock import Mock

from src.data_providers.base import PriceData
from src.data_providers.manager import DataProviderManager
from src.portfolio.manager import PortfolioManager
from src.portfolio.market_data_store import PriceEntry
from src.portfolio.models import Currency, FinancialInstrument, InstrumentType
from src.portfolio.storage import FileBasedStorage


def _make_manager(tmp_dir):
    manager = PortfolioManager(
        storage=Mock(spec=FileBasedStorage),
        data_manager=Mock(spec=DataProviderManager),
        data_dir=tmp_dir,
    )
    manager.current_portfolio = None
    return manager


def _instrument(**kwargs):
    base = dict(
        symbol="CNKY",
        name="iShares Nikkei 225 ETF JPY Acc",
        instrument_type=InstrumentType.ETF,
        currency=Currency.JPY,
    )
    base.update(kwargs)
    return FinancialInstrument(**base)


def _seed(manager, symbol, currency, points):
    entries = [
        PriceEntry(symbol=symbol, date=d, price=Decimal(str(v)),
                   currency=currency, source="market_data")
        for d, v in points
    ]
    manager._market_data_store.set_prices_batch(entries)


def test_scale_guard_rejects_wrong_scale_block():
    """A ~10x drop versus the last known price is rejected, good data preserved."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        inst = _instrument()
        _seed(m, "CNKY", Currency.JPY, [(date(2026, 8, 31), 67890)])

        # Provider returns the wrong (~1/10 scale) Tokyo listing.
        m.data_manager.get_historical_prices = Mock(return_value=[
            PriceData(symbol="CNKY", date=date(2026, 9, 1), close_price=Decimal("6858"),
                      currency=Currency.JPY),
            PriceData(symbol="CNKY", date=date(2026, 9, 2), close_price=Decimal("6659"),
                      currency=Currency.JPY),
        ])

        stored, failed = m._fetch_and_store_prices(
            "CNKY", date(2026, 9, 1), date(2026, 9, 2), inst)

        assert stored == 0
        assert set(failed) == {date(2026, 9, 1), date(2026, 9, 2)}
        # Existing good price untouched.
        assert m._market_data_store.get_latest_price("CNKY") == (date(2026, 8, 31), Decimal("67890"))


def test_scale_guard_allows_normal_moves():
    """Ordinary day-to-day moves within threshold are stored."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        inst = _instrument()
        _seed(m, "CNKY", Currency.JPY, [(date(2026, 8, 31), 67890)])

        m.data_manager.get_historical_prices = Mock(return_value=[
            PriceData(symbol="CNKY", date=date(2026, 9, 1), close_price=Decimal("68580"),
                      currency=Currency.JPY),
            PriceData(symbol="CNKY", date=date(2026, 9, 2), close_price=Decimal("66590"),
                      currency=Currency.JPY),
        ])

        stored, failed = m._fetch_and_store_prices(
            "CNKY", date(2026, 9, 1), date(2026, 9, 2), inst)

        assert stored == 2
        assert failed == []


def test_scale_guard_accepts_first_price_when_no_history():
    """With no prior data the guard cannot anchor, so the price is accepted."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        inst = _instrument()

        m.data_manager.get_historical_prices = Mock(return_value=[
            PriceData(symbol="CNKY", date=date(2026, 9, 1), close_price=Decimal("6858"),
                      currency=Currency.JPY),
        ])

        stored, failed = m._fetch_and_store_prices(
            "CNKY", date(2026, 9, 1), date(2026, 9, 1), inst)

        assert stored == 1
        assert failed == []


def test_currency_guard_rejects_foreign_currency_without_conversion():
    """Provider quoting GBP for a USD-tracked instrument (no price_currency) is skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        inst = _instrument(symbol="HMWO", name="HSBC MSCI World UCITS ETF USD",
                           currency=Currency.USD)
        _seed(m, "HMWO", Currency.USD, [(date(2026, 8, 31), 50)])

        # London line returns GBP (~37) mislabeled; must not be stored as USD.
        m.data_manager.get_historical_prices = Mock(return_value=[
            PriceData(symbol="HMWO", date=date(2026, 9, 1), close_price=Decimal("36.67"),
                      currency=Currency.GBP),
        ])

        stored, failed = m._fetch_and_store_prices(
            "HMWO", date(2026, 9, 1), date(2026, 9, 1), inst)

        assert stored == 0
        assert failed == [date(2026, 9, 1)]
        assert m._market_data_store.get_latest_price("HMWO") == (date(2026, 8, 31), Decimal("50"))


def test_currency_conversion_still_works_when_configured():
    """With price_currency set, provider currency matches and conversion is applied."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        inst = _instrument(symbol="HMWO", name="HSBC MSCI World UCITS ETF USD",
                           currency=Currency.USD, price_currency=Currency.GBP)
        _seed(m, "HMWO", Currency.USD, [(date(2026, 8, 31), 50)])
        m._get_exchange_rate_at_date = Mock(return_value=Decimal("1.372"))

        m.data_manager.get_historical_prices = Mock(return_value=[
            PriceData(symbol="HMWO", date=date(2026, 9, 1), close_price=Decimal("36.67"),
                      currency=Currency.GBP),
        ])

        stored, failed = m._fetch_and_store_prices(
            "HMWO", date(2026, 9, 1), date(2026, 9, 1), inst)

        assert stored == 1
        assert failed == []
        latest = m._market_data_store.get_latest_price("HMWO")
        assert latest[0] == date(2026, 9, 1)
        # 36.67 * 1.372 ~= 50.31, within threshold of the 50 anchor.
        assert abs(float(latest[1]) - 50.31) < 0.1
