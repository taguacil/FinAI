"""Shared state passed between agents in the LangGraph."""

from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AdvisorState(TypedDict, total=False):
    """Mutable state flowing through the advisor graph."""

    # Conversation messages (accumulated across the run)
    messages: Annotated[List[BaseMessage], add_messages]

    # Active portfolio context
    portfolio_name: Optional[str]

    # Per-agent outputs (filled as each node runs)
    research_findings: Optional[str]
    risk_report: Optional[str]
    strategy_proposals: Optional[str]
    recommendations: Optional[str]

    # Free-form user request that triggered this run
    user_request: Optional[str]
