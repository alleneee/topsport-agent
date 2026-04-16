# topsport-agent 概览
- 目标：提供带 ReAct loop 的 agent runtime，支持多 LLM provider、可插拔工具、会话级 memory、skills 和 MCP。
- 技术栈：Python 3.11+，hatchling 构建，pytest/pytest-asyncio 测试，依赖用 uv 管理。
- 代码结构：`src/topsport_agent/engine` 负责运行循环和 hooks；`llm` 负责 provider 抽象和 Anthropic/OpenAI 适配；`memory` 负责文件型会话记忆；`skills` 负责技能注册/加载/激活；`mcp` 负责配置、客户端和工具桥接；`observability` 负责 tracer。