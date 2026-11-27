"""Accrued interest for coupon-paying instruments (bonds, CLNs, reverse convertibles).

A bond's market value is its clean price times the nominal *plus* the interest
earned since the last coupon. Prices in the store are clean, so without this a
bond's value drops by a whole coupon on every payment date and is understated
in between.

Only instruments with stored coupon terms accrue. Notes whose coupon is
conditional (autocallables, barrier memory coupons) should have no terms: the
bank shows no accrued interest for them either.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional

DAY_COUNTS = ("30/360", "ACT/ACT")


@dataclass(frozen=True)
class CouponTerms:
    """Fixed-coupon schedule for one instrument.

    rate: annual coupon in percent (4.15 means 4.15%).
    frequency: coupons per year (1, 2, 4 or 12).
    anchor: any coupon date of the schedule; the day of month is reused for
        every coupon date.
    maturity: last coupon date; nothing accrues on or after it.
    day_count: "30/360" (US bond basis) or "ACT/ACT" (ICMA).
    issue_date: start of the first accrual period, if the instrument was
        issued after the anchor-based period would start.
    """

    rate: Decimal
    frequency: int
    anchor: date
    maturity: date
    day_count: str = "30/360"
    issue_date: Optional[date] = None

    def __post_init__(self):
        if self.frequency not in (1, 2, 4, 12):
            raise ValueError(f"frequency must be 1, 2, 4 or 12, got {self.frequency}")
        if self.day_count not in DAY_COUNTS:
            raise ValueError(f"day_count must be one of {DAY_COUNTS}, got {self.day_count}")


def _add_months(d: date, months: int, day: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _days_30_360(d1: date, d2: date) -> int:
    """30/360 US bond basis."""
    day1 = min(d1.day, 30)
    day2 = 30 if (d2.day == 31 and day1 == 30) else d2.day
    return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (day2 - day1)


def coupon_period(terms: CouponTerms, on: date) -> Optional[tuple[date, date]]:
    """Return (period start, next coupon date) containing ``on``, or None."""
    if on >= terms.maturity:
        return None
    step = 12 // terms.frequency
    day = terms.anchor.day
    # Move from the anchor to the last coupon date on or before `on`.
    months = (on.year - terms.anchor.year) * 12 + (on.month - terms.anchor.month)
    k = months // step
    prev = _add_months(terms.anchor, k * step, day)
    while prev > on:
        k -= 1
        prev = _add_months(terms.anchor, k * step, day)
    nxt = _add_months(terms.anchor, (k + 1) * step, day)
    while nxt <= on:
        k += 1
        prev, nxt = nxt, _add_months(terms.anchor, (k + 1) * step, day)
    if terms.issue_date:
        if on < terms.issue_date:
            return None
        if prev < terms.issue_date:
            prev = terms.issue_date
    return prev, min(nxt, terms.maturity)


def accrued_interest(terms: CouponTerms, nominal: Decimal, on: date) -> Decimal:
    """Interest accrued on ``nominal`` from the last coupon date up to ``on``."""
    if nominal <= 0:
        return Decimal("0")
    period = coupon_period(terms, on)
    if period is None:
        return Decimal("0")
    start, nxt = period
    rate = terms.rate / Decimal(100)
    if terms.day_count == "30/360":
        frac = Decimal(_days_30_360(start, on)) / Decimal(360)
        return nominal * rate * frac
    # ACT/ACT ICMA: actual days over actual days in the regular coupon period.
    step = 12 // terms.frequency
    regular_start = _add_months(nxt, -step, terms.anchor.day)
    period_days = (nxt - regular_start).days
    frac = Decimal((on - start).days) / Decimal(period_days) / Decimal(terms.frequency)
    return nominal * rate * frac
