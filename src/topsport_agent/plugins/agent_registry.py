"""Agent 注册表与工具：解析 agents/*.md，暴露 list_agents/spawn_agent 给 LLM。"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..skills._frontmatter import parse as parse_frontmatter
from ..types.tool import ToolContext, ToolSpec
from .plugin import PluginDescriptor

_logger = logging.getLogger(__name__)


# spawn_agent 执行器签名：传入 AgentDefinition + task 文本 + 调用上下文，返回结果 dict。
# 结果 dict 至少应包含 "text" (最终回复) 和 "ok" (是否成功)。
SpawnExecutor = Callable[
    ["AgentDefinition", str, ToolContext],
    Awaitable[dict[str, Any]],
]


@dataclass(slots=True)
class AgentDefinition:
    """一个 Claude plugin agent 的定义。"""

    name: str
    qualified_name: str
    description: str
    body: str
    allowed_tools: list[str] = field(default_factory=list)
    auto_skills: list[str] = field(default_factory=list)
    model: str = "inherit"
    source_path: Path = field(default_factory=lambda: Path("."))


def _parse_agent_md(path: Path, plugin_name: str) -> AgentDefinition | None:
    """解析 agents/*.md 文件为 AgentDefinition。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    meta, body = parse_frontmatter(text)
    name = meta.get("name", "").strip()
    if not name:
        # 无 name 时从文件名推导
        name = path.stem

    description = meta.get("description", "").strip()
    qualified_name = f"{plugin_name}:{name}"

    # tools 字段：逗号分隔的工具名列表
    raw_tools = meta.get("tools", "").strip()
    allowed_tools = [t.strip() for t in raw_tools.split(",") if t.strip()] if raw_tools else []

    # skills 字段：可能是 YAML list 的简化表示（换行分隔）或逗号分隔
    auto_skills: list[str] = []
    raw_skills = meta.get("skills", "").strip()
    if raw_skills:
        # 处理多行格式 "- skill1\n- skill2" 和单行逗号格式
        for line in raw_skills.replace(",", "\n").splitlines():
            cleaned = line.strip().lstrip("- ").strip()
            if cleaned:
                auto_skills.append(cleaned)

    model = meta.get("model", "inherit").strip()

    return AgentDefinition(
        name=name,
        qualified_name=qualified_name,
        description=description,
        body=body.rstrip("\n"),
        allowed_tools=allowed_tools,
        auto_skills=auto_skills,
        model=model,
        source_path=path,
    )


class AgentRegistry:
    """Agent 注册表：扫描插件 agents/*.md，提供查询接口。同名先注册优先。"""

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def list(self) -> list[AgentDefinition]:
        return sorted(self._agents.values(), key=lambda a: a.qualified_name)

    def register(self, agent: AgentDefinition) -> None:
        self._agents.setdefault(agent.qualified_name, agent)

    def load_from_plugins(self, plugins: list[PluginDescriptor]) -> None:
        """扫描所有插件的 agents/*.md，解析并注册。"""
        for plugin in plugins:
            for agent_path in plugin.agent_paths:
                agent = _parse_agent_md(agent_path, plugin.info.name)
                if agent is None:
                    _logger.debug("skipping agent: %s", agent_path)
                    continue
                self.register(agent)


def build_agent_tools(
    registry: AgentRegistry,
    executor: SpawnExecutor | None = None,
) -> list[ToolSpec]:
    """暴露给 LLM 的 agent 工具：list_agents + spawn_agent。

    executor: 真正创建子 Engine + Session 并执行任务的回调。
              注入 None 时 spawn_agent 退化为只返回 agent 定义（用于测试或预览）。
    """

    async def list_agents(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        agents = registry.list()
        return {
            "count": len(agents),
            "agents": [
                {
                    "name": a.qualified_name,
                    "description": a.description,
                    "model": a.model,
                }
                for a in agents
            ],
        }

    async def spawn_agent(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        name = args.get("name", "")
        task = args.get("task", "")
        agent = registry.get(name)
        if agent is None:
            return {
                "ok": False,
                "error": f"agent '{name}' not found",
                "available": [a.qualified_name for a in registry.list()],
            }

        if executor is None:
            # 回退模式：只返回元信息，让上层（或人工）处理执行
            return {
                "ok": True,
                "executed": False,
                "name": agent.qualified_name,
                "description": agent.description,
                "system_prompt": agent.body,
                "model": agent.model,
                "task": task,
                "allowed_tools": agent.allowed_tools,
                "auto_skills": agent.auto_skills,
            }

        # 真实执行路径：委托给 executor
        try:
            result = await executor(agent, task, ctx)
        except Exception as exc:
            return {
                "ok": False,
                "executed": True,
                "name": agent.qualified_name,
                "error": f"sub-agent crashed: {type(exc).__name__}: {exc}",
            }

        # 保证返回结构有 executed + name
        result.setdefault("executed", True)
        result.setdefault("name", agent.qualified_name)
        return result

    list_spec = ToolSpec(
        name="list_agents",
        description="List all available plugin agents with name, description, and model.",
        parameters={"type": "object", "properties": {}},
        handler=list_agents,
    )

    spawn_spec = ToolSpec(
        name="spawn_agent",
        description=(
            "Spawn a plugin agent to handle a specific task. "
            "The agent runs as an isolated sub-engine with its own session."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Qualified agent name (e.g. 'superpowers:code-reviewer')",
                },
                "task": {
                    "type": "string",
                    "description": "Task description for the agent to execute",
                },
            },
            "required": ["name", "task"],
        },
        handler=spawn_agent,
        required_permissions=frozenset({"agent.spawn"}),
    )

    return [list_spec, spawn_spec]
