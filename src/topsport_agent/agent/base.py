"""Agent 抽象基类：封装 Engine + 扩展能力为单一对象。

参考 Claude Code 的 Agent 模型：
- 一个 Agent 代表一个完整配置好的交互代理（系统提示词 + 工具集 + 能力）
- 子类/工厂方法产出不同预设的 Agent（default / browser / ...）
- Agent 层本身只做组装与生命周期管理，不重复 Engine 的 ReAct 循环逻辑
"""

from __future__ import annotations

import importlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine import Engine, EngineConfig
from ..engine.hooks import ContextProvider, EventSubscriber, PostStepHook, ToolSource
from ..llm.provider import LLMProvider
from ..memory.file_store import FileMemoryStore
from ..memory.injector import MemoryInjector
from ..memory.tools import build_memory_tools
from ..plugins import PluginManager
from ..plugins.agent_registry import AgentDefinition, build_agent_tools
from ..skills import (
    SkillInjector,
    SkillLoader,
    SkillMatcher,
    SkillRegistry,
    build_skill_tools,
)
from ..types.events import Event, EventType
from ..types.message import Message, Role
from ..types.session import RunState, Session
from ..types.tool import ToolContext, ToolSpec


@dataclass(slots=True)
class AgentConfig:
    """Agent 的身份与能力声明。

    name/description 区分不同 Agent；能力开关决定初始化时挂载哪些子系统。
    """

    name: str
    description: str
    system_prompt: str
    model: str
    max_steps: int = 20

    # 能力开关
    enable_skills: bool = True
    enable_memory: bool = True
    enable_plugins: bool = True
    enable_browser: bool = False
    # 启用流式输出（需要 provider 实现 StreamingLLMProvider）
    stream: bool = False

    # 目录配置（只在启用对应能力时生效）
    memory_base_path: Path | None = None
    local_skill_dirs: list[Path] = field(default_factory=list)

    # 自定义扩展
    extra_tools: list[ToolSpec] = field(default_factory=list)
    extra_context_providers: list[ContextProvider] = field(default_factory=list)
    extra_tool_sources: list[ToolSource] = field(default_factory=list)
    extra_post_step_hooks: list[PostStepHook] = field(default_factory=list)
    extra_event_subscribers: list[EventSubscriber] = field(default_factory=list)

    # 其它 engine 级选项
    provider_options: dict[str, Any] | None = None


class Agent:
    """高层代理对象。组装 Engine 及其所有扩展，提供统一的 run/close 接口。"""

    def __init__(
        self,
        provider: LLMProvider,
        config: AgentConfig,
        *,
        engine: Engine,
        cleanup_callbacks: list[Callable[[], Awaitable[None]]] | None = None,
        skill_registry: SkillRegistry | None = None,
        plugin_manager: PluginManager | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._engine = engine
        self._cleanup_callbacks = list(cleanup_callbacks or [])
        self._skill_registry = skill_registry
        self._plugin_manager = plugin_manager

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def skill_registry(self) -> SkillRegistry | None:
        return self._skill_registry

    @property
    def plugin_manager(self) -> PluginManager | None:
        return self._plugin_manager

    def new_session(self, session_id: str | None = None) -> Session:
        """创建一个绑定当前 Agent system_prompt 的新会话。"""
        return Session(
            id=session_id or str(uuid.uuid4()),
            system_prompt=self._config.system_prompt,
        )

    async def run(
        self, user_input: str, session: Session
    ) -> AsyncIterator[Event]:
        """附加一条用户消息到 session，驱动 engine 跑一轮推理。"""
        session.messages.append(Message(role=Role.USER, content=user_input))
        session.state = RunState.IDLE
        async for event in self._engine.run(session):
            yield event
        self._engine.reset_cancel()

    def cancel(self) -> None:
        self._engine.cancel()

    async def close(self) -> None:
        """按注册顺序执行清理回调，吞掉异常避免清理链中断。"""
        for callback in self._cleanup_callbacks:
            try:
                await callback()
            except Exception:
                pass

    # -----------------------------------------------------------------
    # 工厂
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, provider: LLMProvider, config: AgentConfig) -> Agent:
        """按 config 声明组装所有能力并构造 Agent。"""
        tools: list[ToolSpec] = list(config.extra_tools)
        context_providers: list[ContextProvider] = list(config.extra_context_providers)
        tool_sources: list[ToolSource] = list(config.extra_tool_sources)
        post_step_hooks: list[PostStepHook] = list(config.extra_post_step_hooks)
        event_subscribers: list[EventSubscriber] = list(config.extra_event_subscribers)
        cleanup_callbacks: list[Callable[[], Awaitable[None]]] = []

        skill_registry: SkillRegistry | None = None
        plugin_manager: PluginManager | None = None

        # Plugins 先加载，后续 skills 会用到 plugin 提供的 skill_dirs
        if config.enable_plugins:
            plugin_manager = PluginManager()
            plugin_manager.load()
            # 构造 spawn_executor：引用父 provider + 父 tools 快照，使子代理有隔离执行环境
            executor = _build_spawn_executor(provider, config, lambda: list(tools))
            tools.extend(build_agent_tools(plugin_manager.agent_registry(), executor))
            event_subscribers.append(plugin_manager.hook_runner())
            cleanup_callbacks.append(_async_wrap(plugin_manager.cleanup))

        if config.enable_skills:
            plugin_skill_dirs = plugin_manager.skill_dirs() if plugin_manager else []
            all_skill_dirs = list(config.local_skill_dirs) + plugin_skill_dirs
            skill_registry = SkillRegistry(all_skill_dirs)
            skill_registry.load()
            skill_loader = SkillLoader(skill_registry)
            skill_matcher = SkillMatcher(skill_registry)
            tools.extend(build_skill_tools(skill_registry, skill_matcher))
            context_providers.append(
                SkillInjector(skill_registry, skill_loader, skill_matcher)
            )

        if config.enable_memory:
            base = config.memory_base_path or (Path.home() / ".topsport-agent" / "memory")
            memory_store = FileMemoryStore(base)
            tools.extend(build_memory_tools(memory_store))
            context_providers.append(MemoryInjector(memory_store))

        # Browser 延后初始化：playwright 未装或初始化失败时静默跳过
        if config.enable_browser:
            browser_client, browser_sources = _try_make_browser()
            if browser_client is not None:
                tool_sources.extend(browser_sources)
                cleanup_callbacks.append(browser_client.close)

        engine_config = EngineConfig(
            model=config.model,
            max_steps=config.max_steps,
            provider_options=config.provider_options,
            stream=config.stream,
        )
        engine = Engine(
            provider=provider,
            tools=tools,
            config=engine_config,
            context_providers=context_providers,
            tool_sources=tool_sources,
            post_step_hooks=post_step_hooks,
            event_subscribers=event_subscribers,
        )

        return cls(
            provider=provider,
            config=config,
            engine=engine,
            cleanup_callbacks=cleanup_callbacks,
            skill_registry=skill_registry,
            plugin_manager=plugin_manager,
        )


# ---------------------------------------------------------------------------
# 共享辅助
# ---------------------------------------------------------------------------


def _async_wrap(sync_fn: Callable[[], None]) -> Callable[[], Awaitable[None]]:
    """同步函数包为异步 callable，统一清理回调签名。"""

    async def wrapper() -> None:
        sync_fn()

    return wrapper


def _try_make_browser() -> tuple[Any, list[ToolSource]]:
    """尝试构造 BrowserClient 和对应 tool source。未装 playwright 返回 (None, [])。"""
    try:
        importlib.import_module("playwright.async_api")
    except ImportError:
        return None, []
    try:
        from ..browser import BrowserClient, BrowserConfig, BrowserToolSource

        client = BrowserClient.from_config(BrowserConfig(headless=True))
        source = BrowserToolSource(client)
        return client, [source]
    except Exception:
        return None, []


def extract_assistant_text(events: list[Event], session: Session) -> str | None:
    """从 run 事件流中抽取最后一条 assistant 文本。REPL/SDK 取最终回复都用它。"""
    for event in reversed(events):
        if event.type == EventType.MESSAGE_APPENDED:
            if event.payload.get("role") == "assistant":
                last = session.messages[-1] if session.messages else None
                if last and last.role == Role.ASSISTANT and last.content:
                    return last.content
    return None


# ---------------------------------------------------------------------------
# Sub-agent 执行器
# ---------------------------------------------------------------------------


def _build_spawn_executor(
    provider: LLMProvider,
    parent_config: AgentConfig,
    get_parent_tools: Callable[[], list[ToolSpec]],
) -> Callable[[AgentDefinition, str, ToolContext], Awaitable[dict[str, Any]]]:
    """构造 spawn_agent 的真实执行器。

    捕获父 Agent 的 provider 和工具快照（延迟求值 via get_parent_tools，
    这样子代理可见到父 Agent 在 from_config 中最终注册的全部工具）。

    子代理执行策略:
    - model: AgentDefinition.model 若为 "inherit" 则用父 config.model，否则原样使用
    - tools: 若 allowed_tools 非空则按名字过滤父工具集；否则继承父完整工具集
    - system_prompt: 直接使用 AgentDefinition.body
    - auto_skills: 暂未实现（需要子 Engine 有自己的 skill_matcher），记为已知 TODO
    - 子 session 独立，不与父 session 共享消息历史
    """

    async def executor(
        agent: AgentDefinition, task: str, ctx: ToolContext
    ) -> dict[str, Any]:
        model = parent_config.model if agent.model == "inherit" else agent.model

        parent_tools = get_parent_tools()
        if agent.allowed_tools:
            allowed = set(agent.allowed_tools)
            sub_tools = [t for t in parent_tools if t.name in allowed]
        else:
            sub_tools = list(parent_tools)

        sub_engine = Engine(
            provider=provider,
            tools=sub_tools,
            config=EngineConfig(
                model=model,
                max_steps=parent_config.max_steps,
                provider_options=parent_config.provider_options,
            ),
        )
        sub_session = Session(
            id=f"{ctx.session_id}:sub:{agent.qualified_name}:{uuid.uuid4().hex[:6]}",
            system_prompt=agent.body,
        )
        sub_session.messages.append(Message(role=Role.USER, content=task))

        final_text: str | None = None
        tool_call_count = 0
        error: str | None = None

        try:
            async for event in sub_engine.run(sub_session):
                if event.type == EventType.TOOL_CALL_START:
                    tool_call_count += 1
                elif event.type == EventType.ERROR:
                    error = f"{event.payload.get('kind', 'error')}: {event.payload.get('message', '')}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        # 取最后一条 assistant 消息内容作为最终回复
        for msg in reversed(sub_session.messages):
            if msg.role == Role.ASSISTANT and msg.content:
                final_text = msg.content
                break

        if error is not None:
            return {
                "ok": False,
                "executed": True,
                "name": agent.qualified_name,
                "error": error,
                "partial_text": final_text,
            }

        return {
            "ok": True,
            "executed": True,
            "name": agent.qualified_name,
            "text": final_text or "",
            "tool_calls": tool_call_count,
            "messages": len(sub_session.messages),
        }

    return executor
