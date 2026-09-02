"""Unit tests for the canonical asset-class taxonomy.

Pins down the single source of truth so the instrument-type and view-mode maps
can't silently drift from the derived type sets or the consumers that import it.
"""

import pytest

from src.portfolio import asset_classes as ac


@pytest.mark.parametrize(
    "instrument_type,expected",
    [
        ("stock", "equity"),
        ("etf", "equity"),
        ("bond", "fixed_income"),
        ("structured_product", "structured"),
        ("crypto", "other"),
        ("cash", "other"),
        ("", "other"),
        (None, "other"),
    ],
)
def test_category_for_instrument_type(instrument_type, expected):
    assert ac.category_for_instrument_type(instrument_type) == expected


@pytest.mark.parametrize(
    "view_mode,expected",
    [
        ("equities_only", "equity"),
        ("fixed_income_only", "fixed_income"),
        ("structured_only", "structured"),
        ("other_only", "other"),
        ("all", None),
        ("bogus", None),
        (None, None),
    ],
)
def test_category_for_view_mode(view_mode, expected):
    assert ac.category_for_view_mode(view_mode) == expected


def test_type_sets_derive_from_instrument_map():
    """The per-category type sets must be consistent with the instrument map."""
    for types, category in [
        (ac.EQUITY_TYPES, ac.EQUITY),
        (ac.FIXED_INCOME_TYPES, ac.FIXED_INCOME),
        (ac.STRUCTURED_TYPES, ac.STRUCTURED),
    ]:
        for t in types:
            assert ac.category_for_instrument_type(t) == category


def test_consumers_reference_canonical_map():
    """Manager, history and web services must not re-declare the mapping."""
    from src.portfolio.portfolio_history import PortfolioHistory
    from src.web import services

    assert PortfolioHistory.EQUITY_TYPES is ac.EQUITY_TYPES
    assert services._CATEGORY_BY_VIEW_MODE is ac.CATEGORY_BY_VIEW_MODE
    assert services._instrument_category is ac.category_for_instrument_type
