"""Verify the read-only policy blocks every mutating MCP tool."""

from __future__ import annotations

from advisor.config.tool_policy import (
    MARKET_DATA_WRITE_TOOLS,
    READ_TOOLS,
    default_policy,
)

# Tools that exist on the FinAI MCP server but MUST be blocked.
MUTATING_TOOLS = {
    "add_transaction",
    "bulk_add_transactions",
    "modify_transaction",
    "delete_transaction",
    "set_market_price",
    "bulk_set_market_price",
    "set_data_provider_symbol",
    "set_price_currency",
    "ingest_pdf",
}


def test_policy_allows_reads():
    p = default_policy()
    for name in READ_TOOLS:
        assert p.is_allowed(name), name


def test_policy_allows_market_data_refresh():
    p = default_policy()
    for name in MARKET_DATA_WRITE_TOOLS:
        assert p.is_allowed(name), name


def test_policy_blocks_mutations():
    p = default_policy()
    for name in MUTATING_TOOLS:
        assert not p.is_allowed(name), f"policy must block {name}"


def test_filter_removes_blocked_names():
    p = default_policy()
    mixed = ["get_portfolio_summary", "add_transaction", "refresh_data", "delete_transaction"]
    assert p.filter(mixed) == ["get_portfolio_summary", "refresh_data"]


def test_read_and_write_sets_are_disjoint():
    assert READ_TOOLS.isdisjoint(MARKET_DATA_WRITE_TOOLS)
