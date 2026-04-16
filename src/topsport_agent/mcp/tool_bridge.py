from __future__ import annotations

import logging
from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .client import MCPClient

_logger = logging.getLogger(__name__)


class MCPToolSource:
    def __init__(self, client: MCPClient, *, prefix: str | None = None) -> None:
        self._client = client
        # 运行时统一把 MCP 原始工具名包成 <server>.<tool>，避免跨服务重名。
        self._prefix = f"{prefix if prefix is not None else client.name}."

    @property
    def name(self) -> str:
        return self._client.name

    async def list_tools(self) -> list[ToolSpec]:
        try:
            mcp_tools = await self._client.list_tools()
        except Exception as exc:
            _logger.warning(
                "mcp client %r list_tools failed: %r", self._client.name, exc
            )
            return []
        # 桥接层只负责把 MCP 工具描述翻译成运行时 ToolSpec。
        return [self._wrap(tool) for tool in mcp_tools]

    def _wrap(self, mcp_tool: Any) -> ToolSpec:
        raw_name = getattr(mcp_tool, "name", "")
        description = getattr(mcp_tool, "description", None) or ""
        schema = getattr(mcp_tool, "inputSchema", None) or {"type": "object"}
        return ToolSpec(
            name=f"{self._prefix}{raw_name}",
            description=description,
            parameters=schema,
            handler=self._make_handler(raw_name),
        )

    def _make_handler(self, raw_name: str):
        client = self._client

        async def handler(
            args: dict[str, Any], ctx: ToolContext
        ) -> dict[str, Any]:
            # 这里保留运行时工具返回结构，屏蔽底层 MCP 返回对象差异。
            try:
                result = await client.call_tool(raw_name, args)
            except Exception as exc:
                return {
                    "is_error": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "text": "",
                    "structured": None,
                }

            text_parts: list[str] = []
            for content in getattr(result, "content", None) or []:
                text = getattr(content, "text", None)
                if text is not None:
                    text_parts.append(text)

            return {
                "is_error": bool(getattr(result, "isError", False)),
                "text": "\n".join(text_parts),
                "structured": getattr(result, "structuredContent", None),
            }

        return handler
