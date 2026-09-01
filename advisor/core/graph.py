"""LangGraph wiring: orchestrator → research → risk → strategy → recommender.

Phase 1 (skeleton): linear pipeline. Conditional routing comes later.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from advisor.agents import (
    orchestrator,
    recommender_agent,
    research_agent,
    risk_agent,
    strategy_agent,
)
from advisor.agents.base import BaseAdvisorAgent
from advisor.config.settings import AdvisorSettings
from advisor.core.state import AdvisorState


def _run_agent_node(agent: BaseAdvisorAgent, output_key: str):
    """Build a node function that runs an agent once and stores its text output."""

    llm = agent.build_llm()
    tool_node = ToolNode(list(agent.tools)) if agent.tools else None

    async def node(state: AdvisorState) -> dict:
        prior = state.get("user_request") or ""
        upstream = _format_upstream(state)
        messages = [
            SystemMessage(content=agent.system_prompt),
            HumanMessage(content=f"User request:\n{prior}\n\nUpstream context:\n{upstream}"),
        ]
        response = await llm.ainvoke(messages)

        # Phase-1 stub: if the agent emits tool calls, execute them once and
        # feed results back for a final response. Real ReAct loop comes later.
        if tool_node and getattr(response, "tool_calls", None):
            tool_messages = await tool_node.ainvoke({"messages": [response]})
            follow_up = await llm.ainvoke(messages + [response] + tool_messages["messages"])
            text = follow_up.content
        else:
            text = response.content

        return {output_key: text}

    return node


def _format_upstream(state: AdvisorState) -> str:
    parts = []
    for key, label in (
        ("research_findings", "Research"),
        ("risk_report", "Risk"),
        ("strategy_proposals", "Strategy"),
    ):
        val = state.get(key)
        if val:
            parts.append(f"### {label}\n{val}")
    return "\n\n".join(parts) if parts else "(none yet)"


def build_graph(settings: AdvisorSettings, tools: List[BaseTool]):
    """Compile the advisor LangGraph using the given tool set."""

    research = research_agent.make(settings.worker_model, tools, settings.temperature)
    risk = risk_agent.make(settings.worker_model, tools, settings.temperature)
    strategy = strategy_agent.make(settings.worker_model, tools, settings.temperature)
    recommender = recommender_agent.make(settings.worker_model, tools, settings.temperature)
    # orchestrator currently unused in linear flow; kept here for future routing.
    _ = orchestrator.make(settings.orchestrator_model, settings.temperature)

    g = StateGraph(AdvisorState)
    g.add_node("research", _run_agent_node(research, "research_findings"))
    g.add_node("risk", _run_agent_node(risk, "risk_report"))
    g.add_node("strategy", _run_agent_node(strategy, "strategy_proposals"))
    g.add_node("recommender", _run_agent_node(recommender, "recommendations"))

    g.set_entry_point("research")
    g.add_edge("research", "risk")
    g.add_edge("risk", "strategy")
    g.add_edge("strategy", "recommender")
    g.add_edge("recommender", END)

    return g.compile()
