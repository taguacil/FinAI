"""Thin async wrapper around the MCP stdio client for FinAI."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool

from advisor.config.settings import AdvisorSettings

logger = logging.getLogger(__name__)


class FinAIMCPClient:
    """Owns the MCP stdio session for the duration of a console run.

    Use as an async context manager:
        async with FinAIMCPClient(settings) as client:
            tools = await client.list_tools()
            result = await client.call_tool("get_portfolio_summary", {})
    """

    def __init__(self, settings: AdvisorSettings) -> None:
        self._settings = settings
        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    async def __aenter__(self) -> "FinAIMCPClient":
        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self._settings.mcp_command,
            args=self._settings.mcp_args,
            env=None,
            cwd=str(self._settings.project_root),
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("MCP session initialized")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP session not initialized — use as async context manager")
        return self._session

    async def list_tools(self) -> List[MCPTool]:
        result = await self.session.list_tools()
        return list(result.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self.session.call_tool(name, arguments)
        return result
