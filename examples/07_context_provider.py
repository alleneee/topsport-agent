"""示例 07：ContextProvider —— 每步注入临时上下文。

ContextProvider 的输出是 **ephemeral**：只参与本步 LLM 调用，不落盘到 session.messages。
下一步会再次调用 provide() 重新生成（见 loop.py:_collect_ephemeral_context 以及
CLAUDE.md 中的 "Ephemeral context must not persist into session.messages" 不变量）。

典型用法：动态系统提示、RAG 检索结果、用户画像快照、时间戳等。

运行：
    uv sync --group llm
    ANTHROPIC_API_KEY=sk-... uv run python examples/07_context_provider.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from topsport_agent.agent import Agent, AgentConfig
from topsport_agent.llm.providers.anthropic import AnthropicProvider
from topsport_agent.types.message import Message, Role
from topsport_agent.types.session import Session


class ClockContext:
    """每步注入当前时间作为 system 段。
    Role.SYSTEM 消息会被 PromptBuilder 合并进单一 system 块，避免多系统段。
    """

    name = "clock-context"

    async def provide(self, session: Session) -> list[Message]:
        now = datetime.now().isoformat(timespec="seconds")
        return [
            Message(
                role=Role.SYSTEM,
                content=f"Current local time: {now}",
                extra={"section_tag": "clock", "section_priority": 300},
            )
        ]


async def main() -> None:
    provider = AnthropicProvider()
    agent = Agent.from_config(
        provider,
        AgentConfig(
            name="clock-agent",
            description="Agent with a ClockContext provider",
            system_prompt="You are a helpful assistant. Consider the injected time.",
            model="claude-sonnet-4-6",
            enable_skills=False,
            enable_memory=False,
            enable_plugins=False,
            enable_browser=False,
            extra_context_providers=[ClockContext()],
        ),
    )
    try:
        session = agent.new_session()
        async for _ in agent.run("现在大概是什么时段？请直接回答。", session):
            pass
        print("ASSISTANT:", session.messages[-1].content)
        # 关键观察：session.messages 中不会出现 ClockContext 注入的 system 段
        system_count = sum(1 for m in session.messages if m.role == Role.SYSTEM)
        print(f"system messages persisted: {system_count}  (应该是 0)")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
