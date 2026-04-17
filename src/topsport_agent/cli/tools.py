"""CLI 内置工具：用于验证引擎工具调用链路。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..types.tool import ToolContext, ToolSpec


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, str]:
    return {"echoed": args.get("text", "")}


async def _time_handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, str]:
    now = datetime.now(UTC)
    return {"utc": now.isoformat(), "unix": str(int(now.timestamp()))}


async def _calc_handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, str]:
    expr = args.get("expression", "")
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expr):
        return {"error": "only digits and +-*/.() allowed"}
    try:
        result = eval(expr)  # noqa: S307 — restricted to arithmetic chars
    except Exception as exc:
        return {"error": str(exc)}
    return {"expression": expr, "result": str(result)}


def builtin_tools() -> list[ToolSpec]:
    """返回 CLI 模式的内置工具列表。"""
    return [
        ToolSpec(
            name="echo",
            description="Echo back the given text. Useful for testing tool calls.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=_echo_handler,
        ),
        ToolSpec(
            name="current_time",
            description="Return the current UTC time.",
            parameters={"type": "object", "properties": {}},
            handler=_time_handler,
        ),
        ToolSpec(
            name="calc",
            description="Evaluate a simple arithmetic expression (digits and +-*/.() only).",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=_calc_handler,
        ),
    ]
