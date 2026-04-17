"""示例 06：流式输出。

stream=True 时，引擎会把 provider 的 text_delta 以 LLM_TEXT_DELTA 事件暴露出来。
前提：provider 实现了 StreamingLLMProvider（AnthropicProvider / OpenAIChatProvider 都支持）。
流式聚合后的最终 response 仍会经由 parse_response，
确保 session.messages 与非流式路径结构一致。

运行：
    uv sync --group llm
    ANTHROPIC_API_KEY=sk-... uv run python examples/06_streaming.py
"""

from __future__ import annotations

import asyncio

from topsport_agent.agent import default_agent
from topsport_agent.llm.providers.anthropic import AnthropicProvider
from topsport_agent.types.events import EventType


async def main() -> None:
    provider = AnthropicProvider()
    agent = default_agent(
        provider,
        model="claude-sonnet-4-6",
        enable_browser=False,
        stream=True,
    )

    try:
        session = agent.new_session()
        async for event in agent.run("用一句话解释 CAP 定理。", session):
            if event.type == EventType.LLM_TEXT_DELTA:
                print(event.payload["delta"], end="", flush=True)
        print()  # 补一个换行
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
