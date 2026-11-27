"""Accrued interest, checked against the bank's 25.11.2026 wealth statement."""

import tempfile
from datetime import date
from decimal import Decimal

import pytest

from src.portfolio.accrued_interest import CouponTerms, accrued_interest, coupon_period
from src.portfolio.market_data_store import MarketDataStore


def T(rate, freq, anchor, maturity, dc="30/360", issue=None):
    return CouponTerms(Decimal(rate), freq, anchor, maturity, dc, issue)


ON = date(2026, 11, 25)


@pytest.mark.parametrize("terms, nominal, expected", [
    # 30/360 semi-annual US corporates
    (T("4.15", 2, date(2025, 11, 15), date(2030, 11, 15)), 25000, "28.82"),   # Broadcom
    (T("4.55", 2, date(2025, 10, 15), date(2029, 10, 15)), 25000, "126.39"),  # HPE
    (T("4.742", 2, date(2025, 9, 16), date(2032, 3, 16)), 25000, "227.22"),   # BAT
    # ACT/ACT annual EUR
    (T("3.25", 1, date(2026, 1, 10), date(2032, 1, 10), "ACT/ACT"), 50000, "1420.21"),  # Mercedes
    # annual CLN and pay-at-maturity reverse convertible
    (T("6.20", 1, date(2026, 3, 7), date(2030, 3, 7)), 25000, "1110.83"),
    (T("10.60", 1, date(2026, 4, 27), date(2027, 4, 27)), 20000, "1224.89"),
])
def test_matches_bank_statement(terms, nominal, expected):
    got = accrued_interest(terms, Decimal(nominal), ON)
    assert got.quantize(Decimal("0.01")) == Decimal(expected)


def test_resets_to_zero_on_coupon_date_and_grows_after():
    t = T("4.15", 2, date(2025, 11, 15), date(2030, 11, 15))
    assert accrued_interest(t, Decimal(25000), date(2026, 11, 15)) == 0
    assert accrued_interest(t, Decimal(25000), date(2026, 11, 14)) > Decimal("500")  # nearly a full coupon
    assert accrued_interest(t, Decimal(25000), date(2026, 11, 16)) > 0


def test_nothing_before_issue_or_after_maturity():
    t = T("14.88", 4, date(2026, 9, 3), date(2027, 6, 4), issue=date(2026, 9, 3))
    assert accrued_interest(t, Decimal(20000), date(2026, 9, 1)) == 0
    assert accrued_interest(t, Decimal(20000), date(2027, 6, 4)) == 0
    assert accrued_interest(t, Decimal(20000), date(2027, 6, 2)) > 0  # 3 Jun is a coupon date


def test_period_from_issue_date_when_issued_mid_period():
    t = T("4.00", 2, date(2026, 1, 16), date(2031, 1, 16), issue=date(2026, 3, 1))
    assert coupon_period(t, date(2026, 4, 1)) == (date(2026, 3, 1), date(2026, 7, 16))


def test_month_end_anchor_is_clamped():
    t = T("5.00", 12, date(2026, 1, 31), date(2027, 1, 31))
    assert coupon_period(t, date(2026, 2, 15)) == (date(2026, 1, 31), date(2026, 2, 28))


def test_invalid_terms_rejected():
    with pytest.raises(ValueError):
        T("5", 3, date(2026, 1, 1), date(2027, 1, 1))
    with pytest.raises(ValueError):
        T("5", 2, date(2026, 1, 1), date(2027, 1, 1), dc="ACT/365")


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        st = MarketDataStore(tmp)
        t = T("3.25", 1, date(2026, 1, 10), date(2032, 1, 10), "ACT/ACT")
        st.set_coupon_terms("mercedes_2032", t, "note")
        assert st.get_coupon_terms() == {"MERCEDES_2032": t}
        st.delete_coupon_terms("MERCEDES_2032")
        assert st.get_coupon_terms() == {}
