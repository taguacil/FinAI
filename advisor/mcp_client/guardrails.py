"""Policy guard + JSONL audit logger for MCP tool calls."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

from mcp.types import Tool as MCPTool

from advisor.config.tool_policy import ToolPolicy

logger = logging.getLogger(__name__)


def filter_tools(tools: Iterable[MCPTool], policy: ToolPolicy) -> List[MCPTool]:
    """Drop any tool not on the allowlist."""
    kept = [t for t in tools if policy.is_allowed(t.name)]
    dropped = [t.name for t in tools if not policy.is_allowed(t.name)]
    if dropped:
        logger.info("Policy blocked %d tools: %s", len(dropped), sorted(dropped))
    return kept


class AuditLogger:
    """Append-only JSONL log of every tool call attempted by the advisor."""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = log_dir / f"session-{stamp}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        tool: str,
        arguments: dict[str, Any],
        allowed: bool,
        error: str | None = None,
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "arguments": arguments,
            "allowed": allowed,
            "error": error,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
