"""示例 01：最小可用 Engine + 自定义 ToolSpec。

演示核心：
- ToolSpec 构造（name / description / JSON Schema parameters / async handler）
- 直接使用 Engine 的裸 API（不经过 Agent 封装层）
- 事件流消费

运行：
    uv sync --group llm
    ANTHROPIC_API_KEY=sk-... uv run python examples/01_minimal_tool.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from topsport_agent.engine import Engine, EngineConfig
from topsport_agent.llm.providers.anthropic import AnthropicProvider
from topsport_agent.types.message import Message, Role
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


async def add_handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return {"sum": int(args["a"]) + int(args["b"])}


add_tool = ToolSpec(
    name="add",
    description="Add two integers and return their sum.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
    handler=add_handler,
)


async def main() -> None:
    provider = AnthropicProvider()
    engine = Engine(
        provider=provider,
        tools=[add_tool],
        config=EngineConfig(model="claude-sonnet-4-6", max_steps=5),
    )

    session = Session(id="demo-01", system_prompt="You are a calculator.")
    session.messages.append(
        Message(role=Role.USER, content="What is 17 + 25? Use the add tool.")
    )

    async for event in engine.run(session):
        print(f"[{event.type.value}] {event.payload}")

    print("\nFINAL:", session.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
