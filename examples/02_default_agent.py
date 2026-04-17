"""示例 02：开箱即用的默认 Agent。

default_agent() 一次性挂上：
- file_ops（read_file / write_file / edit_file / list_dir / glob_files / grep_files）
- skills（list_skills / load_skill / unload_skill + SkillInjector 上下文注入）
- memory（save_memory / recall_memory / forget_memory + MemoryInjector）
- plugins（list_agents / spawn_agent，若有已安装的 agent 插件）
- browser（可选，playwright 未装时自动跳过）

运行：
    uv sync --group llm
    ANTHROPIC_API_KEY=sk-... uv run python examples/02_default_agent.py
"""

from __future__ import annotations

import asyncio

from topsport_agent.agent import default_agent
from topsport_agent.llm.providers.anthropic import AnthropicProvider


async def main() -> None:
    provider = AnthropicProvider()
    agent = default_agent(
        provider=provider,
        model="claude-sonnet-4-6",
        enable_browser=False,  # 没装 playwright 时务必关掉，避免不必要的初始化
        stream=False,
    )
    try:
        session = agent.new_session()
        async for event in agent.run(
            "用 list_dir 列出 /tmp 下前 5 个文件，然后总结你看到的内容。", session
        ):
            # 只打印关键节点，避免噪音
            if event.type.value in {"tool_call_start", "tool_call_end", "llm_call_end"}:
                print(f"[{event.type.value}] {event.payload}")

        print("\nASSISTANT:", session.messages[-1].content)
    finally:
        # close() 负责卸载 plugins / 关闭 browser 等资源
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
