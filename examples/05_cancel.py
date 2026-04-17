"""示例 05：取消在途推理。

Engine.cancel() 通过 asyncio.Event 触发，
_call_llm_with_cancel 用 asyncio.wait(FIRST_COMPLETED) 让 LLM 调用立即被中断，
不依赖轮询（见 loop.py:_call_llm_with_cancel）。

运行：
    uv sync --group llm
    ANTHROPIC_API_KEY=sk-... uv run python examples/05_cancel.py
"""

from __future__ import annotations

import asyncio

from topsport_agent.agent import default_agent
from topsport_agent.llm.providers.anthropic import AnthropicProvider
from topsport_agent.types.events import EventType


async def main() -> None:
    provider = AnthropicProvider()
    agent = default_agent(provider, "claude-sonnet-4-6", enable_browser=False)

    try:
        session = agent.new_session()

        async def runner() -> None:
            async for event in agent.run("写一首 500 行的长诗。", session):
                if event.type == EventType.CANCELLED:
                    print("[cancelled]")

        task = asyncio.create_task(runner())
        await asyncio.sleep(2.0)
        print("→ 发送 cancel")
        agent.cancel()
        await task

        print(f"\nfinal state: {session.state.value}")
        print(f"messages stored: {len(session.messages)}")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
