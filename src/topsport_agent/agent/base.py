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
from typing import TYPE_CHECKING, Any

from ..engine import Engine, EngineConfig
from ..engine.hooks import ContextProvider, EventSubscriber, PostStepHook, ToolSource
from ..engine.sanitizer import ToolResultSanitizer
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

if TYPE_CHECKING:
    from ..engine.permission.audit import AuditLogger
    from ..engine.permission.filter import ToolVisibilityFilter
    from ..engine.permission.persona_registry import PersonaRegistry
    from ..types.permission import (
        Persona,
        PermissionAsker,
        PermissionChecker,
    )


@dataclass(slots=True)
class AgentConfig:
    """Agent 的身份与能力声明。

    name/description 区分不同 Agent；能力开关决定初始化时挂载哪些子系统。
    """

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    model: str = ""
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

    # Prompt injection 防御：None 表示禁用（Engine 对 untrusted 工具结果不做消毒）。
    # 非 None 时 Engine 对 untrusted 工具结果做消毒并注入 security guard。
    sanitizer: ToolResultSanitizer | None = None

    # 其它 engine 级选项
    provider_options: dict[str, Any] | None = None

    # Permission wiring (optional). When `persona` is set, Agent.new_session_async
    # resolves it and copies permissions into the new Session.
    persona: "Persona | str | None" = None
    persona_registry: "PersonaRegistry | None" = None
    tenant_id: str | None = None

    # v2 capability-ACL hooks. Propagated to Engine as-is; Engine interprets None
    # as "disabled" for each, preserving back-compat.
    permission_filter: "ToolVisibilityFilter | None" = None
    audit_logger: "AuditLogger | None" = None
    permission_checker: "PermissionChecker | None" = None
    permission_asker: "PermissionAsker | None" = None


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
        capability_bundle: dict[str, Any] | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._engine = engine
        self._cleanup_callbacks = list(cleanup_callbacks or [])
        self._skill_registry = skill_registry
        self._plugin_manager = plugin_manager
        # H-A2：父 Agent 构造期的能力快照（tools / providers / sources / hooks /
        # subscribers），spawn_child 要拿来给子代理用，实现能力 parity。
        self._capability_bundle: dict[str, Any] = capability_bundle or {}

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

    async def new_session_async(
        self, session_id: str | None = None,
    ) -> Session:
        """Async session factory that resolves persona → granted_permissions.

        Use this when AgentConfig.persona is set. The synchronous new_session
        still works for callers that don't need permission wiring.
        """
        # Runtime import for isinstance check; TYPE_CHECKING import above only
        # covers type hints.
        from ..types.permission import Persona as _Persona

        session = self.new_session(session_id)
        cfg = self._config
        if cfg.tenant_id is not None:
            session.tenant_id = cfg.tenant_id
        persona_obj: _Persona | None = None
        if isinstance(cfg.persona, _Persona):
            persona_obj = cfg.persona
        elif isinstance(cfg.persona, str) and cfg.persona_registry is not None:
            persona_obj = await cfg.persona_registry.get(cfg.persona)
        if persona_obj is not None:
            session.granted_permissions = persona_obj.permissions
            session.persona_id = persona_obj.id
        return session

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
            # spawn_executor 需要访问构造完成后的父 Agent 实例；先用占位符引用，
            # from_config 末尾回填真正的 Agent。
            parent_ref: list[Agent] = []

            def _parent_getter() -> Agent:
                if not parent_ref:
                    raise RuntimeError(
                        "spawn_agent invoked before parent Agent finished construction"
                    )
                return parent_ref[0]

            executor = _build_spawn_executor(_parent_getter)
            tools.extend(build_agent_tools(plugin_manager.agent_registry(), executor))
            event_subscribers.append(plugin_manager.hook_runner())
            cleanup_callbacks.append(_async_wrap(plugin_manager.cleanup))
        else:
            parent_ref = []

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
            sanitizer=config.sanitizer,
            permission_filter=config.permission_filter,
            audit_logger=config.audit_logger,
            permission_checker=config.permission_checker,
            permission_asker=config.permission_asker,
        )

        bundle: dict[str, Any] = {
            "tools": list(tools),
            "context_providers": list(context_providers),
            "tool_sources": list(tool_sources),
            "post_step_hooks": list(post_step_hooks),
            "event_subscribers": list(event_subscribers),
            "sanitizer": config.sanitizer,
            # v2 capability-ACL: sub-agents must inherit the parent's permission
            # surface. Missing any of these would let delegation paths bypass
            # the ACL — which is exactly the scenario enterprise ACLs must cover.
            "permission_filter": config.permission_filter,
            "audit_logger": config.audit_logger,
            "permission_checker": config.permission_checker,
            "permission_asker": config.permission_asker,
        }

        agent = cls(
            provider=provider,
            config=config,
            engine=engine,
            cleanup_callbacks=cleanup_callbacks,
            skill_registry=skill_registry,
            plugin_manager=plugin_manager,
            capability_bundle=bundle,
        )
        # 回填占位符，spawn_child 现在拿得到完整能力
        parent_ref.append(agent)
        return agent

    # -----------------------------------------------------------------
    # Sub-agent (spawn_child) — H-A2 capability parity
    # -----------------------------------------------------------------

    async def spawn_child(
        self,
        *,
        model: str,
        system_prompt: str,
        task: str,
        allowed_tool_names: list[str] | None = None,
        session_id_prefix: str = "sub",
        parent_session: Session | None = None,
    ) -> tuple[Session, Engine]:
        """构造一个继承父 Agent 全部能力的子代理 Engine + Session。

        继承内容（by reference，共享单例）：
        - context_providers（skills / memory injector / 任何 extra）
        - tool_sources（MCP 桥接 / browser）
        - post_step_hooks（compaction 等）
        - event_subscribers（Langfuse / metrics / plugin hooks）
        - permission_filter / audit_logger / permission_checker / permission_asker

        parent_session：若提供，子 session 继承其 tenant_id / principal /
        granted_permissions / persona_id。未提供则回退到父 engine 的当前运行
        session（spawn_agent 工具在父 engine.run 里调用时始终可用）。
        无父 session 时子代理的 granted_permissions=∅，若父侧配了 filter 会
        filter 空工具集 —— 这是安全的 fail-closed 语义。

        仅 tools 可按 allowed_tool_names 收窄；model 可覆盖；system_prompt 子代理自定。
        调用方自行驱动 engine.run(session) 并收集事件。
        """
        parent_tools: list[ToolSpec] = self._capability_bundle.get("tools", [])
        if allowed_tool_names is not None:
            allow = set(allowed_tool_names)
            sub_tools = [t for t in parent_tools if t.name in allow]
        else:
            sub_tools = list(parent_tools)

        sub_engine = Engine(
            provider=self._provider,
            tools=sub_tools,
            config=EngineConfig(
                model=model,
                max_steps=self._config.max_steps,
                provider_options=self._config.provider_options,
            ),
            # H-A2 关键：把所有非 tool 的能力 by reference 传给子 Engine
            context_providers=list(
                self._capability_bundle.get("context_providers", [])
            ),
            tool_sources=list(self._capability_bundle.get("tool_sources", [])),
            post_step_hooks=list(self._capability_bundle.get("post_step_hooks", [])),
            event_subscribers=list(
                self._capability_bundle.get("event_subscribers", [])
            ),
            sanitizer=self._capability_bundle.get("sanitizer"),
            # v2 capability-ACL parity：permission filter + audit + checker/asker
            # 必须和父代理一致，否则 delegation 路径可以绕过企业 ACL。
            permission_filter=self._capability_bundle.get("permission_filter"),
            audit_logger=self._capability_bundle.get("audit_logger"),
            permission_checker=self._capability_bundle.get("permission_checker"),
            permission_asker=self._capability_bundle.get("permission_asker"),
        )
        sub_session = Session(
            id=f"{session_id_prefix}:{uuid.uuid4().hex[:8]}",
            system_prompt=system_prompt,
        )
        # Capability / tenant inheritance. Fall back to parent engine's current
        # running session (set by Engine._run_inner) when caller didn't pass one
        # — e.g. the default spawn_agent executor invokes spawn_child from
        # within the parent run, where self._engine._current_session is live.
        inherited_from = parent_session or self._engine._current_session
        if inherited_from is not None:
            sub_session.tenant_id = inherited_from.tenant_id
            sub_session.principal = inherited_from.principal
            sub_session.granted_permissions = inherited_from.granted_permissions
            sub_session.persona_id = inherited_from.persona_id
        sub_session.messages.append(Message(role=Role.USER, content=task))
        return sub_session, sub_engine


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
    get_parent_agent: Callable[[], Agent],
) -> Callable[[AgentDefinition, str, ToolContext], Awaitable[dict[str, Any]]]:
    """构造 spawn_agent 的真实执行器。

    H-A2 关键变化：子代理通过 parent.spawn_child() 继承父 Agent 的**全部**能力
    （context_providers / tool_sources / post_step_hooks / event_subscribers）。
    之前版本只继承 tools，子代理丢了 skills/memory/compaction/tracing，三个
    reviewer 独立发现的问题。

    get_parent_agent 是迟绑：构造阶段父 Agent 还在拼装，使用阶段它已完整可用。

    子代理执行策略:
    - model: AgentDefinition.model 若为 "inherit" 则用父 config.model，否则原样使用
    - tools: 若 allowed_tools 非空则按名字过滤父工具集；否则继承父完整工具集
    - system_prompt: 直接使用 AgentDefinition.body
    - 子 session 独立，不与父 session 共享消息历史
    """

    async def executor(
        agent: AgentDefinition, task: str, ctx: ToolContext
    ) -> dict[str, Any]:
        parent = get_parent_agent()
        model = (
            parent.config.model if agent.model == "inherit" else agent.model
        )

        sub_session, sub_engine = await parent.spawn_child(
            model=model,
            system_prompt=agent.body,
            task=task,
            allowed_tool_names=(
                list(agent.allowed_tools) if agent.allowed_tools else None
            ),
            session_id_prefix=f"{ctx.session_id}:sub:{agent.qualified_name}",
        )

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
