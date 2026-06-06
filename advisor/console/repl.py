"""Interactive REPL for the advisor."""

from __future__ import annotations

import logging
from typing import Callable, Dict

from langchain_core.messages import HumanMessage

from advisor.config.settings import AdvisorSettings
from advisor.config.tool_policy import default_policy
from advisor.console.render import Renderer
from advisor.core.graph import build_graph
from advisor.core.memory import SessionMemory
from advisor.core.state import AdvisorState
from advisor.mcp_client.client import FinAIMCPClient
from advisor.mcp_client.guardrails import AuditLogger, filter_tools
from advisor.mcp_client.tool_adapter import mcp_tools_as_langchain

logger = logging.getLogger(__name__)


class Repl:
    def __init__(self, settings: AdvisorSettings) -> None:
        self.settings = settings
        self.renderer = Renderer()
        self.policy = default_policy()
        self.audit = AuditLogger(settings.log_dir)
        self.memory = SessionMemory(settings.log_dir)
        self.portfolio_name: str | None = settings.portfolio_name

    async def run(self) -> None:
        r = self.renderer
        r.banner("FinAI Advisor", "Read-only agentic advisor — type /help for commands")

        async with FinAIMCPClient(self.settings) as client:
            mcp_tools = await client.list_tools()
            allowed = filter_tools(mcp_tools, self.policy)
            dropped_names = {t.name for t in mcp_tools} - {t.name for t in allowed}
            r.tools_table([t.name for t in allowed], dropped_names)

            lc_tools = mcp_tools_as_langchain(
                allowed, client=client, policy=self.policy, audit=self.audit
            )
            r.info(f"Audit log: {self.audit.path}")
            r.info(f"Session memory: {self.memory.path}")

            graph = build_graph(self.settings, lc_tools)
            handlers = self._command_handlers(client, graph)

            while True:
                try:
                    line = input("\nadvisor> ").strip()
                except (EOFError, KeyboardInterrupt):
                    r.info("bye")
                    return
                if not line:
                    continue
                if line.startswith("/"):
                    cmd, *rest = line[1:].split(maxsplit=1)
                    arg = rest[0] if rest else ""
                    handler = handlers.get(cmd)
                    if not handler:
                        r.warn(f"unknown command: /{cmd}")
                        continue
                    try:
                        await handler(arg)
                    except Exception as e:  # surface errors but keep REPL alive
                        r.error(f"{type(e).__name__}: {e}")
                    if cmd == "exit":
                        return
                    continue
                # Free-text → analyze
                await self._analyze(graph, line)

    def _command_handlers(self, client: FinAIMCPClient, graph) -> Dict[str, Callable]:
        r = self.renderer

        async def help_(_: str) -> None:
            r.section(
                "commands",
                "- `/portfolios` — list portfolios\n"
                "- `/portfolio <name>` — select portfolio\n"
                "- `/refresh` — refresh market data\n"
                "- `/analyze` — run full analysis on current portfolio\n"
                "- `/exit` — quit\n"
                "- free text — treated as an analysis request",
            )

        async def portfolios(_: str) -> None:
            result = await client.call_tool("list_portfolios", {})
            r.section("portfolios", _flatten(result))

        async def portfolio(arg: str) -> None:
            if not arg:
                r.warn("usage: /portfolio <name>")
                return
            result = await client.call_tool("select_portfolio", {"portfolio_name": arg})
            self.portfolio_name = arg
            r.section(f"portfolio: {arg}", _flatten(result))

        async def refresh(_: str) -> None:
            result = await client.call_tool("refresh_data", {})
            r.section("refresh_data", _flatten(result))

        async def analyze(arg: str) -> None:
            await self._analyze(graph, arg or "Run a full portfolio analysis and recommend trades.")

        async def exit_(_: str) -> None:
            r.info("bye")

        return {
            "help": help_,
            "portfolios": portfolios,
            "portfolio": portfolio,
            "refresh": refresh,
            "analyze": analyze,
            "exit": exit_,
            "quit": exit_,
        }

    async def _analyze(self, graph, user_request: str) -> None:
        r = self.renderer
        state: AdvisorState = {
            "messages": [HumanMessage(content=user_request)],
            "user_request": user_request,
            "portfolio_name": self.portfolio_name,
        }
        r.info("running advisor graph (research → risk → strategy → recommender)…")
        final = await graph.ainvoke(state)

        for key, label in (
            ("research_findings", "Research"),
            ("risk_report", "Risk"),
            ("strategy_proposals", "Strategy"),
            ("recommendations", "Recommendations"),
        ):
            r.section(label, final.get(key) or "(empty)")

        self.memory.append(
            "analysis",
            {
                "user_request": user_request,
                "portfolio_name": self.portfolio_name,
                "recommendations": final.get("recommendations"),
            },
        )


def _flatten(result) -> str:
    if hasattr(result, "content") and result.content:
        return "\n".join(getattr(b, "text", "") for b in result.content if getattr(b, "text", None))
    return str(result)
