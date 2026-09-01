"""Audit logger + filter behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace

from advisor.config.tool_policy import default_policy
from advisor.mcp_client.guardrails import AuditLogger, filter_tools


def _fake_mcp_tool(name: str):
    return SimpleNamespace(name=name, description="", inputSchema={"type": "object"})


def test_filter_tools_drops_blocked(tmp_path):
    tools = [_fake_mcp_tool("get_portfolio_summary"), _fake_mcp_tool("add_transaction")]
    kept = filter_tools(tools, default_policy())
    assert [t.name for t in kept] == ["get_portfolio_summary"]


def test_audit_logger_appends_jsonl(tmp_path):
    audit = AuditLogger(tmp_path)
    audit.record("get_portfolio_summary", {"x": 1}, allowed=True)
    audit.record("add_transaction", {"y": 2}, allowed=False, error="policy")

    lines = audit.path.read_text().strip().splitlines()
    assert len(lines) == 2
    a = json.loads(lines[0])
    b = json.loads(lines[1])
    assert a["tool"] == "get_portfolio_summary" and a["allowed"] is True
    assert b["allowed"] is False and b["error"] == "policy"
