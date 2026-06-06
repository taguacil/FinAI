"""Research agent — gathers portfolio + market facts.

Phase 1 (skeleton): minimal toolset, stub behavior.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.tools import BaseTool

from advisor.agents.base import BaseAdvisorAgent

SYSTEM_PROMPT = """You are the research agent.

Your job is to gather facts about the portfolio and market:
  - Portfolio composition, positions, recent transactions
  - Current prices, recent price history, FX rates
  - Data freshness — if stale, refresh before reasoning

Return a compact factual summary. Do NOT recommend trades — that is the
recommender's job.
"""

RESEARCH_TOOLS = {
    "get_portfolio_summary",
    "get_portfolio_snapshot",
    "get_current_price",
    "get_batch_prices",
    "get_price_history",
    "get_fx_rate",
    "get_data_freshness",
    "check_market_data_availability",
    "get_transactions",
    "get_historical_instruments",
    "list_portfolios",
    "select_portfolio",
    "search_instrument",
    "search_company",
    "resolve_instrument",
    "refresh_data",
    "fetch_and_update_prices",
}


def make(
    model_key: str, all_tools: Sequence[BaseTool], temperature: float = 0.1
) -> BaseAdvisorAgent:
    tools = [t for t in all_tools if t.name in RESEARCH_TOOLS]
    return BaseAdvisorAgent(
        name="research",
        system_prompt=SYSTEM_PROMPT,
        model_key=model_key,
        temperature=temperature,
        tools=tools,
    )


ResearchAgent = BaseAdvisorAgent
