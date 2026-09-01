"""Read-only tool policy.

Defines exactly which FinAI MCP tools the advisor is allowed to call.
Anything not on the allowlist is blocked at adapter time so the LLM
can never invoke it. This is the single source of truth for safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

# Tools that read portfolio / market data and do not mutate any persisted state.
READ_TOOLS: Set[str] = {
    # Portfolio reads
    "get_portfolio_summary",
    "get_portfolio_snapshot",
    "get_portfolio_metrics",
    "get_ytd_performance",
    "get_transactions",
    "get_transaction_history",
    "get_historical_instruments",
    "list_portfolios",
    "select_portfolio",
    # Market data reads
    "get_current_price",
    "get_batch_prices",
    "get_price_history",
    "get_fx_rate",
    "get_moving_average_signal",
    "get_data_freshness",
    "check_market_data_availability",
    # Discovery
    "search_instrument",
    "search_company",
    "resolve_instrument",
    # Analytics. These do not mutate portfolio or market data, but the
    # simulation tools below persist an audit record of each run under
    # data/simulations/ (via _record_simulation) so results can be re-checked
    # later. That append-only log is the only side effect; it never changes
    # portfolio holdings, transactions, or prices.
    "optimize_portfolio",
    "simulate_what_if",
    "advanced_what_if",
    "scenario_optimization",
    "test_hypothetical_position",
    "calculator",
}

# Tools that ONLY refresh / fetch market data caches. No portfolio state change.
MARKET_DATA_WRITE_TOOLS: Set[str] = {
    "refresh_data",
    "fetch_and_update_prices",
    "fetch_historical_fx_rates",
    "update_historical_market_data",
    "interpolate_prices",
}


@dataclass(frozen=True)
class ToolPolicy:
    """Allowlist policy. Anything outside `allowed` is blocked."""

    allowed: frozenset[str] = field(
        default_factory=lambda: frozenset(READ_TOOLS | MARKET_DATA_WRITE_TOOLS)
    )

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed

    def filter(self, tool_names) -> list[str]:
        return [t for t in tool_names if self.is_allowed(t)]


def default_policy() -> ToolPolicy:
    return ToolPolicy()
