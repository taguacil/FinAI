"""Tool adapter converts MCP tools into LangChain BaseTools with policy."""

from __future__ import annotations

from types import SimpleNamespace

from advisor.config.tool_policy import default_policy
from advisor.mcp_client.guardrails import AuditLogger
from advisor.mcp_client.tool_adapter import mcp_tools_as_langchain


def _fake_mcp_tool(name: str, schema=None):
    return SimpleNamespace(
        name=name,
        description=f"desc for {name}",
        inputSchema=schema or {"type": "object", "properties": {"symbol": {"type": "string"}}},
    )


class _DummyClient:
    pass


def test_adapter_emits_one_tool_per_allowed_mcp_tool(tmp_path):
    audit = AuditLogger(tmp_path)
    mcp_tools = [
        _fake_mcp_tool("get_portfolio_summary"),
        _fake_mcp_tool("add_transaction"),  # blocked
        _fake_mcp_tool("get_current_price"),
    ]
    lc = mcp_tools_as_langchain(
        mcp_tools, client=_DummyClient(), policy=default_policy(), audit=audit
    )
    names = sorted(t.name for t in lc)
    assert names == ["get_current_price", "get_portfolio_summary"]
    # Each tool exposes an args schema with the declared property.
    for t in lc:
        assert "symbol" in t.args_schema.model_fields
