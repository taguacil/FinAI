"""Canonical asset-class taxonomy.

Single source of truth for mapping instrument types and analytics view modes to
the four portfolio categories (equity, fixed_income, structured, other). The
portfolio-history/manager layer and the web services all import from here so the
mapping can never drift across copies.
"""

from typing import Dict, Optional

# Category keys
EQUITY = "equity"
FIXED_INCOME = "fixed_income"
STRUCTURED = "structured"
OTHER = "other"

# instrument_type -> category. Anything unlisted falls through to OTHER
# (crypto, options, futures, cash, mutual funds, ...).
INSTRUMENT_TYPE_CATEGORY: Dict[str, str] = {
    "stock": EQUITY,
    "etf": EQUITY,
    "bond": FIXED_INCOME,
    # Structured products are their own class — kept distinct from plain bonds
    # so fixed-income analytics aren't polluted by equity-linked risk.
    "structured_product": STRUCTURED,
}

# Instrument types per category (derived; used for value-history filtering).
EQUITY_TYPES = frozenset(t for t, c in INSTRUMENT_TYPE_CATEGORY.items() if c == EQUITY)
FIXED_INCOME_TYPES = frozenset(
    t for t, c in INSTRUMENT_TYPE_CATEGORY.items() if c == FIXED_INCOME
)
STRUCTURED_TYPES = frozenset(
    t for t, c in INSTRUMENT_TYPE_CATEGORY.items() if c == STRUCTURED
)

# Analytics/dashboard view mode -> category. "all" has no single category.
CATEGORY_BY_VIEW_MODE: Dict[str, str] = {
    "equities_only": EQUITY,
    "fixed_income_only": FIXED_INCOME,
    "structured_only": STRUCTURED,
    "other_only": OTHER,
}


def category_for_instrument_type(instrument_type: Optional[str]) -> str:
    """Map an instrument type to its asset-class category."""
    return INSTRUMENT_TYPE_CATEGORY.get(instrument_type or "", OTHER)


def category_for_view_mode(view_mode: Optional[str]) -> Optional[str]:
    """Map an analytics view mode to a category, or None for "all"/unknown."""
    return CATEGORY_BY_VIEW_MODE.get(view_mode or "")
