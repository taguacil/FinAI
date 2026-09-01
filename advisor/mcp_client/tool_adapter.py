"""Adapt MCP tools into LangChain `BaseTool`s, with policy enforcement."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, ConfigDict, create_model

from advisor.config.tool_policy import ToolPolicy
from advisor.mcp_client.client import FinAIMCPClient
from advisor.mcp_client.guardrails import AuditLogger

logger = logging.getLogger(__name__)


# ---- JSON-Schema → pydantic args model ---------------------------------------

_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _args_model_from_schema(name: str, schema: dict[str, Any] | None) -> Type[BaseModel]:
    """Build a permissive pydantic model from the MCP tool's input schema.

    We keep it loose (all optional, Any) — the MCP server validates the real schema.
    The model is mostly so LangChain can introspect arg names.
    """
    if not schema or schema.get("type") != "object":
        return create_model(
            f"{name}_Args",
            __config__=ConfigDict(extra="allow"),
        )

    properties: dict[str, Any] = schema.get("properties", {}) or {}
    fields: dict[str, tuple[Any, Any]] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _JSON_TYPE_MAP.get(prop_schema.get("type", "string"), Any)
        fields[prop_name] = (Optional[py_type], None)

    return create_model(
        f"{name}_Args",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


# ---- LangChain BaseTool wrapper ----------------------------------------------


class MCPLangChainTool(BaseTool):
    """LangChain tool that proxies a single MCP tool through the policy guard."""

    name: str
    description: str
    args_schema: Type[BaseModel]

    # private (excluded from pydantic validation)
    _client: FinAIMCPClient
    _policy: ToolPolicy
    _audit: AuditLogger

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        client: FinAIMCPClient,
        policy: ToolPolicy,
        audit: AuditLogger,
    ) -> None:
        super().__init__(
            name=name, description=description, args_schema=args_schema
        )
        # Bypass pydantic to set private refs
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_audit", audit)

    def _run(self, **kwargs: Any) -> Any:
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> Any:
        if not self._policy.is_allowed(self.name):
            self._audit.record(self.name, kwargs, allowed=False, error="policy")
            raise PermissionError(f"Tool '{self.name}' blocked by advisor policy")
        try:
            result = await self._client.call_tool(self.name, kwargs)
        except Exception as e:
            self._audit.record(self.name, kwargs, allowed=True, error=str(e))
            raise
        self._audit.record(self.name, kwargs, allowed=True)
        return _serialize_result(result)


def _serialize_result(result: Any) -> str:
    """Flatten an MCP CallToolResult into a plain string for the LLM."""
    if hasattr(result, "content") and result.content:
        parts: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(result)


# ---- Public factory ----------------------------------------------------------


def mcp_tools_as_langchain(
    mcp_tools: List[MCPTool],
    *,
    client: FinAIMCPClient,
    policy: ToolPolicy,
    audit: AuditLogger,
) -> List[BaseTool]:
    """Convert allowed MCP tools into LangChain BaseTools."""
    out: List[BaseTool] = []
    for t in mcp_tools:
        if not policy.is_allowed(t.name):
            continue
        args_model = _args_model_from_schema(t.name, t.inputSchema)
        out.append(
            MCPLangChainTool(
                name=t.name,
                description=t.description or t.name,
                args_schema=args_model,
                client=client,
                policy=policy,
                audit=audit,
            )
        )
    return out
