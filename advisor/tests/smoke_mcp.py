"""Standalone smoke test: launches the real FinAI MCP server over stdio.

Not part of pytest (it spawns a subprocess + reads real portfolio files).
Run manually:  uv run python -m advisor.tests.smoke_mcp
"""

from __future__ import annotations

import asyncio

from advisor.config.settings import load_settings
from advisor.config.tool_policy import default_policy
from advisor.mcp_client.client import FinAIMCPClient
from advisor.mcp_client.guardrails import AuditLogger, filter_tools
from advisor.mcp_client.tool_adapter import mcp_tools_as_langchain


async def main() -> None:
    settings = load_settings()
    policy = default_policy()
    audit = AuditLogger(settings.log_dir)

    async with FinAIMCPClient(settings) as client:
        all_tools = await client.list_tools()
        allowed = filter_tools(all_tools, policy)
        print(f"server exposed {len(all_tools)} tools, policy allows {len(allowed)}")

        lc_tools = mcp_tools_as_langchain(allowed, client=client, policy=policy, audit=audit)
        names = sorted(t.name for t in lc_tools)
        print(f"LangChain tools built: {len(lc_tools)}")
        for n in names[:5]:
            print(f"  - {n}")

        result = await client.call_tool("list_portfolios", {})
        text = "\n".join(
            getattr(b, "text", "") for b in result.content if getattr(b, "text", None)
        )
        print("list_portfolios result (truncated):")
        print(text[:400])


if __name__ == "__main__":
    asyncio.run(main())
