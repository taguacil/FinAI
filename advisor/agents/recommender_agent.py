"""Recommender — final BUY / SELL / HOLD synthesis."""

from __future__ import annotations

from typing import Sequence

from langchain_core.tools import BaseTool

from advisor.agents.base import BaseAdvisorAgent

SYSTEM_PROMPT = """You are the recommender.

Synthesize the research, risk, and strategy outputs into final, actionable
recommendations. For each item, produce:
  - Action: BUY / SELL / HOLD / REBALANCE / HEDGE
  - Instrument and target size (in % of portfolio or absolute units)
  - Rationale grounded in the findings above (cite specific facts)
  - Confidence (low / medium / high) and time horizon
  - Caveats and what would invalidate the recommendation

Be explicit that this is NOT financial advice and the system does not execute
trades.
"""


def make(model_key: str, all_tools: Sequence[BaseTool], temperature: float = 0.2) -> BaseAdvisorAgent:
    # Recommender does not need tools — it reasons over upstream outputs.
    return BaseAdvisorAgent(
        name="recommender",
        system_prompt=SYSTEM_PROMPT,
        model_key=model_key,
        temperature=temperature,
        tools=(),
    )


RecommenderAgent = BaseAdvisorAgent
