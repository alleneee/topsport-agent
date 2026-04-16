from __future__ import annotations

import asyncio
from typing import Any

from topsport_agent.engine import Engine, EngineConfig
from topsport_agent.llm.providers.anthropic import AnthropicProvider
from topsport_agent.types.message import Message, Role
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


async def echo_handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return {"echoed": args}


echo_tool = ToolSpec(
    name="echo",
    description="Echo back the arguments for testing. Use this when asked to repeat something.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    handler=echo_handler,
)


async def main() -> None:
    provider = AnthropicProvider()

    engine = Engine(
        provider,
        tools=[echo_tool],
        config=EngineConfig(model="MiniMax-M2.7"),
    )

    session = Session(
        id="smoke-1",
        system_prompt="You are a test agent. When asked to echo, use the echo tool.",
    )
    session.messages.append(
        Message(role=Role.USER, content="Please echo back: hello world")
    )

    print("=== Running engine ===\n")
    async for event in engine.run(session):
        print(f"  [{event.type.value}] {event.payload}")

    print("\n=== Messages ===\n")
    for msg in session.messages:
        role = msg.role.value
        if msg.content:
            print(f"  {role}: {msg.content[:200]}")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  {role} -> tool_call: {tc.name}({tc.arguments})")
        if msg.tool_results:
            for tr in msg.tool_results:
                err = " [ERROR]" if tr.is_error else ""
                print(f"  {role} <- tool_result{err}: {tr.output}")

    print(f"\nFinal state: {session.state.value}")


if __name__ == "__main__":
    asyncio.run(main())
