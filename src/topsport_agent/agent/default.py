"""默认 Agent：开箱即用的通用代理，开启 skills/memory/plugins/browser。"""

from __future__ import annotations

from pathlib import Path

from typing import Any

from ..engine.hooks import ToolSource
from ..engine.sanitizer import DefaultSanitizer, ToolResultSanitizer
from ..llm.provider import LLMProvider
from ..types.tool import ToolSpec
from .base import Agent, AgentConfig
from .config_parts import AgentIdentity, CapabilityRegistry, CapabilityToggles

# sentinel：区分"未传 sanitizer"和"显式传 None 关闭"。
_DEFAULT = object()

DEFAULT_SYSTEM_PROMPT = (
    "You are topsport-agent, a versatile assistant with access to skills, "
    "working memory, plugin extensions, file operations, and optional browser control.\n\n"
    "Capabilities:\n"
    "- `read_file` / `write_file` / `edit_file`: work with files on disk (absolute paths required)\n"
    "- `list_dir` / `glob_files` / `grep_files`: explore the filesystem and search contents\n"
    "- `list_skills` / `load_skill` / `unload_skill`: discover and activate skills on demand\n"
    "- `save_memory` / `recall_memory` / `forget_memory`: persist session context\n"
    "- `list_agents` / `spawn_agent`: delegate specialized tasks to plugin agents\n"
    "\n"
    "Structured reasoning: inspect available tools first, then act. "
    "Prefer loading a relevant skill before tackling a new domain. "
    "For file edits, read before editing and use exact string matches for edit_file."
)


def default_agent(
    provider: LLMProvider,
    model: str,
    *,
    name: str = "default",
    description: str = "Default topsport-agent with all standard capabilities",
    system_prompt: str | None = None,
    max_steps: int | None = None,
    enable_browser: bool = True,
    enable_file_ops: bool = True,
    enable_skills: bool = True,
    enable_memory: bool = True,
    enable_plugins: bool = True,
    stream: bool = False,
    memory_base_path: Path | None = None,
    local_skill_dirs: list[Path] | None = None,
    extra_tools: list[ToolSpec] | None = None,
    extra_tool_sources: list[ToolSource] | None = None,
    extra_event_subscribers: list[Any] | None = None,
    sanitizer: Any = _DEFAULT,
) -> Agent:
    """标准 Agent 配置：skills + memory + plugins + file_ops + 可选 browser。

    enable_skills / enable_memory / enable_plugins: 单独可关的能力闸门。
        CLI 默认全开；server 默认全关（secure by default，由 ServerConfig
        控制）。之前版本这三个硬编码为 True，绕过了 ServerConfig 的闸门。
    extra_tool_sources: 运行时扩展工具源（如 MCP / OpenSandbox）；透传给 AgentConfig。
    sanitizer: 省略则默认启用 DefaultSanitizer（对 untrusted 工具结果做 prompt
        injection 防御）；显式传 None 则关闭。
    """
    effective_sanitizer: ToolResultSanitizer | None
    if sanitizer is _DEFAULT:
        effective_sanitizer = DefaultSanitizer()
    else:
        effective_sanitizer = sanitizer

    # Reference impl of the structured AgentConfig.from_parts API: instead of
    # passing 12+ flat keywords, assemble three concern-aligned dataclasses.
    # Identical behavior — `from_parts` populates the same flat fields under
    # the hood — but readers can scan identity / toggles / registry separately.
    identity_kwargs: dict[str, Any] = dict(
        name=name,
        description=description,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        model=model,
    )
    if max_steps is not None:
        identity_kwargs["max_steps"] = max_steps

    config = AgentConfig.from_parts(
        identity=AgentIdentity(**identity_kwargs),
        toggles=CapabilityToggles(
            enable_skills=enable_skills,
            enable_memory=enable_memory,
            enable_plugins=enable_plugins,
            enable_browser=enable_browser,
            enable_file_ops=enable_file_ops,
            stream=stream,
            memory_base_path=memory_base_path,
            local_skill_dirs=(
                local_skill_dirs or [Path.home() / ".claude" / "skills"]
            ),
        ),
        registry=CapabilityRegistry(
            extra_tools=list(extra_tools or []),
            extra_tool_sources=list(extra_tool_sources or []),
            extra_event_subscribers=list(extra_event_subscribers or []),
            sanitizer=effective_sanitizer,
        ),
    )
    return Agent.from_config(provider, config)
