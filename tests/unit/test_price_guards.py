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


def test_unheld_symbol_is_fetched_and_stored():
    """A benchmark like SPY (not a holding) is stored as quoted, not skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        m.data_manager.get_historical_prices = Mock(return_value=[
            PriceData(symbol="SPY", date=date(2026, 9, 2), close_price=Decimal("763.26"),
                      currency=Currency.USD),
            PriceData(symbol="SPY", date=date(2026, 9, 3), close_price=Decimal("771.25"),
                      currency=Currency.USD),
        ])

        stored, failed = m._fetch_and_store_prices("SPY", date(2026, 9, 2), date(2026, 9, 3))

        assert stored == 2 and failed == []
        assert m._market_data_store.get_price("SPY", date(2026, 9, 3)) == Decimal("771.25")


def test_backfill_anchors_to_price_before_window_not_latest():
    """Backfilling Sep must compare against Aug, not a much later stored price."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        inst = _instrument(symbol="IBIT", currency=Currency.USD)
        # Aug 31 = 45, then today's price after a >50% fall = 20.
        _seed(m, "IBIT", Currency.USD, [(date(2026, 8, 31), 45), (date(2026, 11, 26), 20)])
        m.data_manager.get_historical_prices = Mock(return_value=[
            PriceData(symbol="IBIT", date=date(2026, 9, 1), close_price=Decimal("44"),
                      currency=Currency.USD),
            PriceData(symbol="IBIT", date=date(2026, 9, 2), close_price=Decimal("43.5"),
                      currency=Currency.USD),
        ])

        stored, failed = m._fetch_and_store_prices(
            "IBIT", date(2026, 9, 1), date(2026, 9, 2), inst)

        assert stored == 2 and failed == []


# -- provider symbol persistence ---------------------------------------------

from datetime import datetime

from src.portfolio.models import Portfolio, Position, Transaction, TransactionType


def _txn(inst, ttype, qty, price, day):
    return Transaction(
        id=f"t{day}", timestamp=datetime(2026, 1, day), instrument=inst,
        transaction_type=ttype, quantity=Decimal(qty), price=Decimal(price),
        currency=inst.currency,
    )


def test_recalculate_positions_keeps_provider_symbol():
    """Rebuilding positions from transactions must not drop the ticker."""
    inst = _instrument(symbol="EQQQ", currency=Currency.USD)
    p = Portfolio(id="p", name="p", base_currency=Currency.USD, created_at=datetime(2026, 1, 1))
    p.add_transaction(_txn(inst.model_copy(), TransactionType.BUY, "10", "500", 2))
    # As loaded from JSON: the position has its own instrument copy, and only it
    # carries the provider settings.
    p.positions["EQQQ"].instrument = inst.model_copy(
        update={"data_provider_symbol": "EQAC.SW", "price_currency": Currency.USD})

    p.recalculate_positions()

    assert p.positions["EQQQ"].instrument.data_provider_symbol == "EQAC.SW"
    assert p.positions["EQQQ"].instrument.price_currency == Currency.USD


def test_mapping_in_db_survives_rebuild_and_reload():
    """The DB mapping is the source of truth and is re-applied after edits."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        inst = _instrument(symbol="EQQQ", currency=Currency.USD)
        m.current_portfolio = Portfolio(
            id="p", name="p", base_currency=Currency.USD, created_at=datetime(2026, 1, 1))
        m.current_portfolio.add_transaction(_txn(inst.model_copy(), TransactionType.BUY, "10", "500", 2))
        m.current_portfolio.add_transaction(_txn(inst.model_copy(), TransactionType.BUY, "5", "510", 3))

        assert m.set_data_provider_symbol("EQQQ", "eqac.sw")
        assert m._market_data_store.get_instrument_mapping("EQQQ")["data_provider_symbol"] == "EQAC.SW"
        # Written onto every copy, transactions included.
        assert all(t.instrument.data_provider_symbol == "EQAC.SW"
                   for t in m.current_portfolio.transactions)

        # Simulate an old file where the ticker was lost, then a delete+rebuild.
        for t in m.current_portfolio.transactions:
            t.instrument.data_provider_symbol = None
        m.current_portfolio.positions["EQQQ"].instrument.data_provider_symbol = None
        assert m.delete_transaction("t3")
        assert m.current_portfolio.positions["EQQQ"].instrument.data_provider_symbol == "EQAC.SW"


def test_existing_portfolio_ticker_is_migrated_into_db():
    with tempfile.TemporaryDirectory() as tmp:
        m = _make_manager(tmp)
        inst = _instrument(symbol="ASML", currency=Currency.EUR, data_provider_symbol="ASML.AS")
        m.current_portfolio = Portfolio(
            id="p", name="p", base_currency=Currency.USD, created_at=datetime(2026, 1, 1))
        m.current_portfolio.add_transaction(_txn(inst, TransactionType.BUY, "1", "1000", 2))

        m._apply_instrument_mappings()

        assert m._market_data_store.get_instrument_mapping("ASML")["data_provider_symbol"] == "ASML.AS"


def test_store_write_does_not_hide_older_prices():
    """Writing a price for an unloaded symbol must not truncate its history in memory."""
    from src.portfolio.market_data_store import MarketDataStore
    with tempfile.TemporaryDirectory() as tmp:
        MarketDataStore(tmp).set_prices_batch([PriceEntry(symbol="X", date=date(2026, 1, 1), price=Decimal("1"),
                                                         currency=Currency.USD, source="t")])
        st = MarketDataStore(tmp)  # fresh process: nothing cached
        st.set_prices_batch([PriceEntry(symbol="X", date=date(2026, 1, 2), price=Decimal("2"),
                                        currency=Currency.USD, source="t")])
        st.set_price("X", date(2026, 1, 3), Decimal("3"), Currency.USD)
        assert sorted(st.get_prices("X", date(2026, 1, 1), date(2026, 1, 3))) == [
            date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]
