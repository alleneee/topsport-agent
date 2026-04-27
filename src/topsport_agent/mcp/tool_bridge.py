from __future__ import annotations

import logging
from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .client import MCPClient

_logger = logging.getLogger(__name__)


async def _do_call(
    client: MCPClient, raw_name: str, args: dict[str, Any],
) -> dict[str, Any]:
    """Invoke MCP `call_tool` and normalise the response shape.

    Three-layer error model:
      - Transport / SDK exception → handler returns is_error=True with
        the exception text (engine's ToolResult layer adds nothing).
      - MCP isError=True (server-reported tool error) → passed through
        verbatim so the LLM can interpret.
      - Everything else → text + structured payload.
    """
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


class MCPToolSource:
    """桥接层：把远端 MCP 工具描述翻译成引擎 ToolSpec，通过 <server>.<tool> 前缀避免跨服务重名。"""
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
        # MCP 服务器来自不可信第三方（本地/远端皆然），返回内容视为 untrusted，
        # 交给 Engine 的 sanitizer 做 prompt injection 防御。
        return ToolSpec(
            name=f"{self._prefix}{raw_name}",
            description=description,
            parameters=schema,
            handler=self._make_handler(raw_name),
            trust_level="untrusted",
            required_permissions=self._client.permissions,
        )

    def _make_handler(self, raw_name: str):
        """三层错误模型：传输异常 -> handler catch; MCP isError -> 透传给 LLM。

        引擎层 ToolResult.is_error 保留给引擎自身。
        """
        client = self._client

        async def handler(
            args: dict[str, Any], ctx: ToolContext
        ) -> dict[str, Any]:
            # 统一返回结构，屏蔽底层 MCP 返回对象差异。
            # client.call_tool 内部从 client._progress_callback 取缺省回调，
            # 此处不显式传 progress_callback —— 让 manager.set_progress_callback
            # 设的全局 callback 生效。需要 per-call override 的高级用例可在
            # 上游构造工具时改写此 closure。
            #
            # Elicitation 路由：把 ctx.session_id 写到 client 实例字段
            # （ContextVar 跨 SDK task 失效，必须用实例字段）。当 elicitation
            # 启用时，`_call_lock` 把同一 client 的 call_tool 串行化以避免
            # 并发用户的 sid 串扰；elicitation 未启用时 lock 退化为 no-op
            # 等价（contention 微小）。
            elicit_enabled = client._elicitation_handler is not None
            if elicit_enabled:
                async with client._call_lock:
                    client._current_call_session_id = ctx.session_id
                    try:
                        return await _do_call(client, raw_name, args)
                    finally:
                        client._current_call_session_id = None
            return await _do_call(client, raw_name, args)

        return handler
