"""Orchestrator — decides which specialists to invoke."""

from __future__ import annotations

from advisor.agents.base import BaseAdvisorAgent

SYSTEM_PROMPT = """You are the orchestrator for a portfolio advisory system.

Given a user request, decide which specialist agents to invoke and in what order:
  - research   : pulls portfolio snapshot, prices, signals, freshness
  - risk       : analyzes exposure, VaR, drawdown, concentration
  - strategy   : proposes strategies (rebalancing, hedging, thematic tilts)
  - recommender: synthesizes a final BUY / SELL / HOLD recommendation

You NEVER mutate the portfolio. You may trigger market-data refreshes only when
data freshness is stale.

Respond with a short plan listing the agents to call and a one-line reason each.
"""


def make(model_key: str, temperature: float = 0.1) -> BaseAdvisorAgent:
    return BaseAdvisorAgent(
        name="orchestrator",
        system_prompt=SYSTEM_PROMPT,
        model_key=model_key,
        temperature=temperature,
        tools=(),
    )


# Keep the class alias for the package __init__ re-export
OrchestratorAgent = BaseAdvisorAgent
