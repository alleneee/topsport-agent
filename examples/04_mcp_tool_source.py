"""示例 04：动态工具源 —— 挂 MCP Server。

通过 MCPManager.from_config_file 读取 Claude Desktop 风格配置文件，
每个 server 会被包成 MCPToolSource（动态工具源）。

MCP 工具名会被自动加 "<server>." 前缀；与 builtin 同名时 builtin 总是胜出
（见 loop.py:_snapshot_tools）。

前置：
    uv sync --group llm --group mcp

config 文件示例 (mcp_servers.json)：
    {
      "mcpServers": {
        "fs": {
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
      }
    }

运行：
    ANTHROPIC_API_KEY=sk-... uv run python examples/04_mcp_tool_source.py ./mcp_servers.json
"""

from __future__ import annotations

import asyncio
import sys
from typing import cast

from topsport_agent.agent import Agent, AgentConfig
from topsport_agent.engine.hooks import ToolSource
from topsport_agent.llm.providers.anthropic import AnthropicProvider
from topsport_agent.mcp.manager import MCPManager


def _as_tool_sources(manager: MCPManager) -> list[ToolSource]:
    # Pyright 对 Protocol 的可变属性做严格不变量检查：MCPToolSource.name 是 @property
    # 而 ToolSource.name 是 str 字段，结构上匹配但静态类型不同。运行时完全兼容，
    # 用 cast 显式告知 Pyright。
    return cast(list[ToolSource], list(manager.tool_sources()))


async def main(config_path: str) -> None:
    provider = AnthropicProvider()
    manager = MCPManager.from_config_file(config_path)

    agent = Agent.from_config(
        provider,
        AgentConfig(
            name="mcp-demo",
            description="Agent with MCP servers as dynamic tool sources",
            system_prompt="You have access to remote tools via MCP. Use them when helpful.",
            model="claude-sonnet-4-6",
            enable_skills=False,
            enable_memory=False,
            enable_plugins=False,
            enable_browser=False,
            # 每步调用 list_tools() 实时拉取远端工具列表
            extra_tool_sources=_as_tool_sources(manager),
        ),
    )

    try:
        session = agent.new_session()
        async for event in agent.run("列出你可以调用的工具，并演示使用其中一个。", session):
            if event.type.value in {"tool_call_start", "tool_call_end"}:
                print(f"[{event.type.value}] {event.payload}")

        print("\nASSISTANT:", session.messages[-1].content)
    finally:
        await agent.close()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./mcp_servers.json"
    asyncio.run(main(path))
