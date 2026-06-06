"""Risk agent — concentration, volatility, VaR, drawdown."""

from __future__ import annotations

from typing import Sequence

from langchain_core.tools import BaseTool

from advisor.agents.base import BaseAdvisorAgent

SYSTEM_PROMPT = """You are the risk agent.

Analyze the portfolio's risk profile:
  - Concentration (per asset, per sector, per currency)
  - Volatility, max drawdown, Sharpe / Sortino
  - VaR / CVaR where computable

Flag any breach of common-sense limits (e.g., > 30% single name).
Output a structured risk report. Do not recommend trades.
"""

RISK_TOOLS = {
    "get_portfolio_metrics",
    "get_portfolio_summary",
    "get_ytd_performance",
    "get_price_history",
    "advanced_what_if",
    "calculator",
}


def make(
    model_key: str, all_tools: Sequence[BaseTool], temperature: float = 0.1
) -> BaseAdvisorAgent:
    tools = [t for t in all_tools if t.name in RISK_TOOLS]
    return BaseAdvisorAgent(
        name="risk",
        system_prompt=SYSTEM_PROMPT,
        model_key=model_key,
        temperature=temperature,
        tools=tools,
    )


RiskAgent = BaseAdvisorAgent
