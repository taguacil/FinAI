"""Unit tests for the structured-product name heuristic.

These pin down that plain bonds named "... Note" or "Callable ..." are NOT
misclassified as structured products (which would drop them from fixed-income
analytics and the optimizer), while genuine structured products still resolve.
"""

import pytest

from src.portfolio.instrument_resolver import InstrumentResolver


@pytest.mark.parametrize(
    "name",
    [
        "Barrier Reverse Convertible on AAPL",
        "Autocallable Note",
        "Credit Linked Note",
        "Shark Note",
        "10.5% RC on Tesla",
        "Some Structured Note",  # bare weak "note", no bond guard
    ],
)
def test_structured_product_names(name):
    assert InstrumentResolver._is_structured_product_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "Medium Term Note",
        "Fixed Rate Note 2028",
        "Floating Rate Note",
        "Callable Bond 5%",
        "US Treasury Note 2030",
        "Senior Note 2027",
        "Bayer 2026 Bond",
        "Apple Inc 3.25% 2029",
    ],
)
def test_plain_bond_names_not_structured(name):
    assert InstrumentResolver._is_structured_product_name(name) is False


def test_none_and_empty_names_not_structured():
    assert InstrumentResolver._is_structured_product_name(None) is False
    assert InstrumentResolver._is_structured_product_name("") is False
