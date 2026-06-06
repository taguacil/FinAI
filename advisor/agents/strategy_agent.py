"""Strategy agent — proposes strategies (rebalance, hedge, tilt)."""

from __future__ import annotations

from typing import Sequence

from langchain_core.tools import BaseTool

from advisor.agents.base import BaseAdvisorAgent

SYSTEM_PROMPT = """You are the strategy agent.

Given the research findings and the risk report, propose 1–3 candidate
strategies. Each strategy must include:
  - A clear thesis (why now, what conviction)
  - Concrete actions (which instruments, target weights or sizes)
  - Expected impact (return / risk tradeoff)
  - Key risks and invalidation conditions

Use simulate_what_if / scenario_optimization / optimize_portfolio to back-check
ideas. Do NOT execute any trade — recommendations only.
"""

STRATEGY_TOOLS = {
    "get_portfolio_summary",
    "get_portfolio_metrics",
    "optimize_portfolio",
    "simulate_what_if",
    "advanced_what_if",
    "scenario_optimization",
    "test_hypothetical_position",
    "get_moving_average_signal",
    "get_price_history",
    "calculator",
}


def make(
    model_key: str, all_tools: Sequence[BaseTool], temperature: float = 0.1
) -> BaseAdvisorAgent:
    tools = [t for t in all_tools if t.name in STRATEGY_TOOLS]
    return BaseAdvisorAgent(
        name="strategy",
        system_prompt=SYSTEM_PROMPT,
        model_key=model_key,
        temperature=temperature,
        tools=tools,
    )


StrategyAgent = BaseAdvisorAgent
