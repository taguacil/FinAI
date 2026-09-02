"""Per-instrument-type filtering in PortfolioHistory.

The web asset-class filters are now built per instrument type (stock, etf, bond,
...) so they mirror the asset-allocation breakdown. This exercises the type-level
attribution that backs them: a symbol resolves to one authoritative type (from
its opening BUY), and cash flows can be attributed to an exact type or to the
broader category.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.portfolio.market_data_store import MarketDataStore
from src.portfolio.models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    Portfolio,
    Transaction,
    TransactionType,
)
from src.portfolio.portfolio_history import PortfolioHistory


def _instrument(symbol, itype, currency=Currency.USD):
    return FinancialInstrument(
        symbol=symbol, name=symbol, instrument_type=itype, currency=currency
    )


def _txn(tid, instrument, ttype, qty, price, when, currency=Currency.USD):
    return Transaction(
        id=tid,
        instrument=instrument,
        transaction_type=ttype,
        quantity=Decimal(str(qty)),
        price=Decimal(str(price)),
        timestamp=when,
        currency=currency,
    )


def _fx(from_ccy, to_ccy, *_args, **_kwargs):
    if from_ccy == to_ccy:
        return Decimal("1")
    if from_ccy == Currency.EUR and to_ccy == Currency.USD:
        return Decimal("1.1")
    return None


@pytest.fixture
def portfolio():
    t0 = datetime(2025, 1, 1)
    aapl = _instrument("AAPL", InstrumentType.STOCK)
    msft = _instrument("MSFT", InstrumentType.STOCK)
    spy = _instrument("SPY", InstrumentType.ETF)
    eur = _instrument("EURSTOCK", InstrumentType.STOCK, Currency.EUR)
    txns = [
        _txn("1", aapl, TransactionType.BUY, 100, 150, t0 + timedelta(days=1)),
        _txn("2", msft, TransactionType.BUY, 50, 300, t0 + timedelta(days=2)),
        _txn("3", spy, TransactionType.BUY, 200, 450, t0 + timedelta(days=3)),
        _txn("4", eur, TransactionType.BUY, 100, 100, t0 + timedelta(days=4), Currency.EUR),
        _txn("5", aapl, TransactionType.SELL, 25, 160, t0 + timedelta(days=5)),
    ]
    p = Portfolio(id="p1", name="P", base_currency=Currency.USD)
    p.transactions = txns
    return p


@pytest.fixture
def history(portfolio, tmp_path):
    store = MarketDataStore(data_dir=str(tmp_path))
    return PortfolioHistory(portfolio, store, _fx, _fx)


@pytest.fixture
def window(portfolio):
    dates = [t.timestamp.date() for t in portfolio.transactions]
    return min(dates) - timedelta(days=1), max(dates) + timedelta(days=1)


def test_type_by_symbol_resolves_from_opening_buy(history):
    types = history._type_by_symbol()
    assert types["AAPL"] == "stock"
    assert types["MSFT"] == "stock"
    assert types["SPY"] == "etf"
    assert types["EURSTOCK"] == "stock"


def test_category_by_symbol_maps_types_to_categories(history):
    cats = history._category_by_symbol()
    assert cats["AAPL"] == "equity"
    assert cats["SPY"] == "equity"  # ETF still rolls up into the equity category


def test_cash_flows_for_exact_type_isolate_that_type(history, window):
    start, end = window
    # Only the SPY (ETF) buy: 200 * 450 = 90,000 USD.
    etf = history.get_category_cash_flows_by_day(start, end, instrument_type="etf")
    assert sum(etf.values()) == Decimal("90000")


def test_cash_flows_for_type_are_subset_of_category(history, window):
    start, end = window
    etf = history.get_category_cash_flows_by_day(start, end, instrument_type="etf")
    stock = history.get_category_cash_flows_by_day(start, end, instrument_type="stock")
    equity = history.get_category_cash_flows_by_day(start, end, category="equity")
    # stocks: AAPL 15,000 + MSFT 15,000 + EURSTOCK 10,000*1.1 - AAPL sell 4,000.
    assert sum(stock.values()) == Decimal("37000")
    # The equity category is exactly stocks + ETFs.
    assert sum(equity.values()) == sum(stock.values()) + sum(etf.values())


def test_unknown_type_has_no_flows(history, window):
    start, end = window
    assert history.get_category_cash_flows_by_day(start, end, instrument_type="bond") == {}
