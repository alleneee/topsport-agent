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

from ..engine import Engine, EngineConfig, EngineRunOptions
from ..engine.hooks import (
    ContextProvider,
    EventSubscriber,
    PostStepHook,
    PostToolUseHook,
    PreToolUseHook,
    ToolSource,
)
from ..engine.sanitizer import ToolResultSanitizer
from ..llm.provider import LLMProvider
from ..plugins import PluginManager
from ..plugins.agent_registry import AgentDefinition
from ..skills import SkillRegistry
from ..types.events import Event, EventType
from ..types.message import ContentPart, Message, Role
from ..types.plan import Plan
from ..types.session import RunState, Session
from ..types.tool import ToolContext, ToolSpec
from .capabilities import CapabilityBundle, CapabilityModule
from .config_parts import (
    DEFAULT_MAX_STEPS,
    AgentIdentity,
    CapabilityRegistry,
    CapabilityToggles,
    identity_field_names,
    isolate_value,
    registry_field_names,
    toggle_field_names,
)

if TYPE_CHECKING:
    from ..engine.checkpoint import Checkpointer
    from ..engine.permission.audit import AuditLogger
    from ..engine.permission.filter import ToolVisibilityFilter
    from ..engine.permission.persona_registry import PersonaRegistry
    from ..llm.image_generation import (
        ImageGenerationResponse,
        OpenAIImageGenerationClient,
    )
    from ..types.permission import (
        PermissionAsker,
        PermissionChecker,
        Persona,
    )


@dataclass(slots=True)
class AgentConfig:
    """Agent 的身份 + 能力声明 + 注册表（合一入口）。

    Phase 4 后这个类被划分成三个概念组（identity / toggles / registry），
    通过 `config.identity` / `.toggles` / `.registry` 属性产出对应的子
    dataclass 视图（`AgentIdentity` / `CapabilityToggles` /
    `CapabilityRegistry`），方便 mock、testing、operator 阅读。
    构造路径仍兼容原 flat keyword 形式；新代码可通过
    `AgentConfig.from_parts(identity=..., toggles=..., registry=...)`
    走结构化路径。flat 字段是当前 PR 的存储真理来源；后续 PR 再考虑反转。
    """

    # ── Identity (see AgentIdentity) ──────────────────────────────────
    name: str = ""
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    max_steps: int = DEFAULT_MAX_STEPS

    # ── Toggles (see CapabilityToggles) ───────────────────────────────
    enable_skills: bool = True
    enable_memory: bool = True
    enable_plugins: bool = True
    enable_browser: bool = False
    # file_ops: read_file / write_file / edit_file / list_dir / glob_files /
    # grep_files. Previously only opt-in via default_agent's enable_file_ops param;
    # now a first-class AgentConfig flag so CapabilityModule assembly is uniform.
    enable_file_ops: bool = False
    # 启用流式输出（需要 provider 实现 StreamingLLMProvider）
    stream: bool = False
    # 目录配置（只在启用对应能力时生效）
    memory_base_path: Path | None = None
    local_skill_dirs: list[Path] = field(default_factory=list)

    # ── Registry: extras (see CapabilityRegistry) ─────────────────────
    extra_tools: list[ToolSpec] = field(default_factory=list)
    extra_context_providers: list[ContextProvider] = field(default_factory=list)
    extra_tool_sources: list[ToolSource] = field(default_factory=list)
    extra_post_step_hooks: list[PostStepHook] = field(default_factory=list)
    extra_event_subscribers: list[EventSubscriber] = field(default_factory=list)
    # PreToolUseHook / PostToolUseHook：Tool 调用前/后的扩展点，对标 Claude Code
    # 的 PreToolUse / PostToolUse hook（拒绝/改写/审计）。
    extra_pre_tool_hooks: list[PreToolUseHook] = field(default_factory=list)
    extra_post_tool_hooks: list[PostToolUseHook] = field(default_factory=list)
    # 第三方 / 应用级 CapabilityModule。Agent.from_config 把它们追加到默认列表
    # 之后，再走 order_capability_modules() 拓扑排序——`depends_on` 可以引用任何
    # 默认 module 名（"plugins" / "skills" / "memory" / "file_ops" / "browser"）
    # 来精确插队，不必 fork default_capability_modules() 也不必 monkey-patch。
    extra_capability_modules: list["CapabilityModule"] = field(default_factory=list)

    # ── Registry: permission / audit ──────────────────────────────────
    persona: "Persona | str | None" = None
    persona_registry: "PersonaRegistry | None" = None
    tenant_id: str | None = None
    permission_filter: "ToolVisibilityFilter | None" = None
    audit_logger: "AuditLogger | None" = None
    permission_checker: "PermissionChecker | None" = None
    permission_asker: "PermissionAsker | None" = None

    # ── Registry: engine-level singletons ─────────────────────────────
    # Prompt injection 防御：None 表示禁用（Engine 对 untrusted 工具结果不做消毒）。
    # 非 None 时 Engine 对 untrusted 工具结果做消毒并注入 security guard。
    sanitizer: ToolResultSanitizer | None = None
    # 其它 engine 级选项
    provider_options: dict[str, Any] | None = None
    # -----------------------------------------------------------------
    # Structured views — read-only snapshots assembled from flat fields.
    # Three guarantees:
    #   1. Each call returns a fresh sub-dataclass instance (no caching).
    #   2. Sub-dataclasses are frozen — mutating scalars on the view
    #      raises FrozenInstanceError instead of silently no-oping.
    #   3. Mutable container fields (list / dict) are shallow-copied via
    #      isolate_value, so list mutations on the view don't write through
    #      to AgentConfig and vice versa. Operators get snapshot semantics
    #      end-to-end; nested objects (ToolSpec, hook instances) remain
    #      shared by reference, matching Python's general "don't deep-copy
    #      unless asked" principle.
    # -----------------------------------------------------------------

    @property
    def identity(self) -> AgentIdentity:
        return AgentIdentity(
            **{n: isolate_value(getattr(self, n)) for n in identity_field_names()}
        )

    @property
    def toggles(self) -> CapabilityToggles:
        return CapabilityToggles(
            **{n: isolate_value(getattr(self, n)) for n in toggle_field_names()}
        )

    @property
    def registry(self) -> CapabilityRegistry:
        return CapabilityRegistry(
            **{n: isolate_value(getattr(self, n)) for n in registry_field_names()}
        )

    @classmethod
    def from_parts(
        cls,
        *,
        identity: AgentIdentity | None = None,
        toggles: CapabilityToggles | None = None,
        registry: CapabilityRegistry | None = None,
    ) -> "AgentConfig":
        """Construct an AgentConfig from structured sub-dataclass parts.

        Any part not provided uses its dataclass defaults. Mutable fields
        (lists/dicts) are shallow-copied so subsequent mutation on the
        provided parts doesn't leak into the constructed AgentConfig and
        vice versa — same snapshot guarantee as the view properties.
        """
        ident = identity if identity is not None else AgentIdentity()
        togs = toggles if toggles is not None else CapabilityToggles()
        reg = registry if registry is not None else CapabilityRegistry()

        kwargs: dict[str, Any] = {}
        for n in identity_field_names():
            kwargs[n] = isolate_value(getattr(ident, n))
        for n in toggle_field_names():
            kwargs[n] = isolate_value(getattr(togs, n))
        for n in registry_field_names():
            kwargs[n] = isolate_value(getattr(reg, n))
        return cls(**kwargs)


@dataclass(slots=True)
class AgentRuntime:
    """Runtime services attached to an Agent instance.

    `AgentConfig` describes what the agent is and which capabilities are
    enabled. `AgentRuntime` carries live process dependencies that should not
    be serialized as config or compared as identity.
    """

    image_generator: OpenAIImageGenerationClient | None = None
    plan_checkpointer: Checkpointer | None = None


class Agent:
    """高层代理对象。组装 Engine 及其所有扩展，提供统一的 run/close 接口。"""

    def __init__(
        self,
        provider: LLMProvider,
        config: AgentConfig,
        *,
        engine: Engine,
        cleanup_callbacks: list[Callable[[], Awaitable[None]]] | None = None,
        capability_bundle: CapabilityBundle | None = None,
        runtime: AgentRuntime | None = None,
        image_generator: OpenAIImageGenerationClient | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._engine = engine
        self._cleanup_callbacks = list(cleanup_callbacks or [])
        # H-A2：父 Agent 构造期的能力快照（tools / providers / sources / hooks /
        # subscribers），spawn_child 直接读 bundle 字段把它们 by-reference 给子代理。
        # skill_registry / plugin_manager 通过 bundle.state["..."] 派生 property。
        self._bundle: CapabilityBundle = capability_bundle or CapabilityBundle()
        self._runtime = runtime or AgentRuntime(image_generator=image_generator)

    @property
    def runtime(self) -> AgentRuntime:
        return self._runtime

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def skill_registry(self) -> SkillRegistry | None:
        return self._bundle.state.get("skill_registry")

    @property
    def plugin_manager(self) -> PluginManager | None:
        return self._bundle.state.get("plugin_manager")

    # -----------------------------------------------------------------
    # Post-construction capability registration. Always updates BOTH the
    # live engine AND the bundle so spawn_child sees the addition. Server-
    # side decorators (metrics / Langfuse / MCP) used to mutate
    # `_capability_bundle` dict directly — that path is now closed; callers
    # must go through these helpers.
    # -----------------------------------------------------------------

    def add_event_subscriber(self, subscriber: EventSubscriber) -> None:
        self._engine.add_event_subscriber(subscriber)
        self._bundle.event_subscribers.append(subscriber)

    def add_tool_source(self, source: ToolSource) -> None:
        self._engine.add_tool_source(source)
        self._bundle.tool_sources.append(source)

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
        self,
        user_input: str | list[ContentPart] | Message | None = None,
        session: Session | None = None,
        *,
        mode: str = "react",
        plan: Plan | None = None,
        system: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> AsyncIterator[Event]:
        """Unified execution entry — the single public verb for this Agent.

        Mode selection:
          - mode="react" (default): classic ReAct loop driven by user_input.
            user_input accepts: str (plain text), list[ContentPart] (multimodal),
            Message (fully formed), or None (continues existing session without
            a new turn — rare, mostly for resume flows).
          - mode="plan": executes a Plan DAG. Requires `plan=<Plan>`. user_input
            is ignored. Each step spawns a child Agent inheriting all capabilities
            (skills / memory / tools / permission filters / event subscribers).

        Request-level overrides:
          - system: replaces session.system_prompt for this turn onward
            (persists to session; subsequent calls keep it unless overridden
            again). Single place for request-scoped persona switches.

        Plan/Orchestrator is no longer a separate public API — it's an internal
        execution strategy selected via mode="plan".
        """
        if session is None:
            raise ValueError("session is required")
        if system is not None and system != session.system_prompt:
            session.system_prompt = system
        run_options = EngineRunOptions(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            provider_options=provider_options,
        )

        if mode == "plan":
            if plan is None:
                raise ValueError("mode='plan' requires plan=<Plan>")
            async for event in self._run_plan(plan, session, run_options):
                yield event
            return
        if mode != "react":
            raise ValueError(
                f"unknown mode {mode!r}; expected 'react' or 'plan'"
            )

        if user_input is None:
            raise ValueError("mode='react' requires user_input")
        if isinstance(user_input, Message):
            msg = user_input
        elif isinstance(user_input, list):
            msg = Message(role=Role.USER, content_parts=user_input)
        else:
            msg = Message(role=Role.USER, content=user_input)
        session.messages.append(msg)
        session.state = RunState.IDLE
        async for event in self._engine.run(session, run_options=run_options):
            yield event
        self._engine.reset_cancel()

    async def _run_plan(
        self,
        plan: Plan,
        session: Session,
        run_options: EngineRunOptions,
    ) -> AsyncIterator[Event]:
        """Internal: execute a Plan via the Orchestrator strategy.

        session's tenant_id / principal / granted_permissions propagate to every
        sub-step via spawn_child (handled by Orchestrator.parent_agent path).
        This method is NOT exposed as a public API — clients drive plans through
        `agent.run(mode="plan", plan=...)`.
        """
        # Lazy import avoids circular (Orchestrator imports Agent for typing).
        from ..engine.orchestrator import Orchestrator, SubAgentConfig

        sub_config = SubAgentConfig(
            provider=self._provider,
            model=self._config.model,
            max_steps=self._config.max_steps,
            provider_options=self._config.provider_options,
        )
        orchestrator = Orchestrator(
            plan,
            sub_config,
            parent_agent=self,
            parent_session=session,
            run_options=run_options,
            checkpointer=self._runtime.plan_checkpointer,
        )
        async for event in orchestrator.execute():
            yield event

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResponse:
        """Generate an image via the configured image_generator.

        Raises RuntimeError if no image_generator was wired into this Agent.
        """
        image_generator = self._runtime.image_generator
        if image_generator is None:
            raise RuntimeError(
                "No image_generator configured; pass AgentRuntime(image_generator=...) "
                "to Agent.from_config(), or construct OpenAIImageGenerationClient directly."
            )
        from ..llm.image_generation import ImageGenerationRequest
        resolved_model = model or image_generator.default_model
        if resolved_model is None:
            raise ValueError(
                "model required (or set default_model on image_generator)"
            )
        request = ImageGenerationRequest(
            prompt=prompt, model=resolved_model, **kwargs
        )
        return await image_generator.generate(request)

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
    def from_config(
        cls,
        provider: LLMProvider,
        config: AgentConfig,
        *,
        runtime: AgentRuntime | None = None,
    ) -> Agent:
        """按 config 声明组装所有能力并构造 Agent.

        Capability assembly goes through the CapabilityModule protocol: each
        capability (plugins / skills / memory / file_ops / browser) is a
        module that decides if it applies and produces a CapabilityBundle.
        Adding a new capability = one new module class in capability_impls.py,
        zero edits to this method.
        """
        from .capabilities import InstallContext, order_capability_modules
        from .capability_impls import default_capability_modules

        # Single aggregator. Pre-seeded with the operator's `extra_*` lists,
        # then merged with each enabled module's bundle. List fields keep
        # registration order (extras first, modules in default order).
        agg = CapabilityBundle(
            tools=list(config.extra_tools),
            context_providers=list(config.extra_context_providers),
            tool_sources=list(config.extra_tool_sources),
            post_step_hooks=list(config.extra_post_step_hooks),
            event_subscribers=list(config.extra_event_subscribers),
            pre_tool_hooks=list(config.extra_pre_tool_hooks),
            post_tool_hooks=list(config.extra_post_tool_hooks),
        )

        # parent_ref is a late-binding slot: modules that build spawn executors
        # (PluginsModule) capture it now, dereference on tool-call time — by
        # which point we've appended the constructed Agent.
        parent_ref: list[Agent] = []
        ctx = InstallContext(config=config, provider=provider, parent_ref=parent_ref)

        # Topo-sort by `depends_on` (ties broken on registration order).
        # Lets modules consume `ctx.shared[<key>]` published by their
        # declared dependencies without anyone hand-curating list order.
        # `extra_capability_modules` from AgentConfig append after defaults;
        # they can declare depends_on=("memory",) etc. to insert anywhere
        # in the topo order without forking default_capability_modules().
        ordered_modules = order_capability_modules(
            [*default_capability_modules(), *config.extra_capability_modules]
        )
        for module in ordered_modules:
            if not module.is_enabled(ctx):
                continue
            module_bundle = module.install(ctx)
            agg.merge(module_bundle)
            # Published state: available to later modules. (`agg.merge` already
            # folded module_bundle.state into agg.state for downstream consumers
            # like Agent property accessors.)
            ctx.shared.update(module_bundle.state)

        # Scalar permission/sanitizer surface comes from AgentConfig directly
        # (modules don't produce these today). Sub-agents inherit them via
        # spawn_child reading the same bundle fields.
        agg.sanitizer = config.sanitizer
        agg.permission_filter = config.permission_filter
        agg.audit_logger = config.audit_logger
        agg.permission_checker = config.permission_checker
        agg.permission_asker = config.permission_asker

        engine_config = EngineConfig(
            model=config.model,
            max_steps=config.max_steps,
            provider_options=config.provider_options,
            stream=config.stream,
        )
        engine = Engine(
            provider=provider,
            tools=agg.tools,
            config=engine_config,
            context_providers=agg.context_providers,
            tool_sources=agg.tool_sources,
            post_step_hooks=agg.post_step_hooks,
            event_subscribers=agg.event_subscribers,
            pre_tool_hooks=agg.pre_tool_hooks,
            post_tool_hooks=agg.post_tool_hooks,
            sanitizer=agg.sanitizer,
            permission_filter=agg.permission_filter,
            audit_logger=agg.audit_logger,
            permission_checker=agg.permission_checker,
            permission_asker=agg.permission_asker,
        )

        agent = cls(
            provider=provider,
            config=config,
            engine=engine,
            cleanup_callbacks=agg.cleanup_callbacks,
            capability_bundle=agg,
            runtime=runtime or AgentRuntime(),
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
        - pre_tool_hooks / post_tool_hooks（PreToolUse / PostToolUse 链）
        - permission_filter / audit_logger / permission_checker / permission_asker

        AgentConfig.extra_capability_modules 注册的第三方模块通过这条路径间接
        继承——它们的 install 输出在父 Agent 构造时已经合并进 self._bundle，
        这里只是把 bundle 的内容传给子 Engine，子代理不会重跑 module install。

        parent_session：若提供，子 session 继承其 tenant_id / principal /
        granted_permissions / persona_id。未提供则回退到父 engine 的当前运行
        session（spawn_agent 工具在父 engine.run 里调用时始终可用）。
        无父 session 时子代理的 granted_permissions=∅，若父侧配了 filter 会
        filter 空工具集 —— 这是安全的 fail-closed 语义。

        仅 tools 可按 allowed_tool_names 收窄；model 可覆盖；system_prompt 子代理自定。
        调用方自行驱动 engine.run(session) 并收集事件。
        """
        parent_tools: list[ToolSpec] = self._bundle.tools
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
            # H-A2 关键：把所有非 tool 的能力 by reference 传给子 Engine。
            # 静态属性访问替代 dict.get(key, []) —— typo 立刻暴露在类型检查里。
            context_providers=list(self._bundle.context_providers),
            tool_sources=list(self._bundle.tool_sources),
            post_step_hooks=list(self._bundle.post_step_hooks),
            event_subscribers=list(self._bundle.event_subscribers),
            pre_tool_hooks=list(self._bundle.pre_tool_hooks),
            post_tool_hooks=list(self._bundle.post_tool_hooks),
            sanitizer=self._bundle.sanitizer,
            # v2 capability-ACL parity：permission filter + audit + checker/asker
            # 必须和父代理一致，否则 delegation 路径可以绕过企业 ACL。
            permission_filter=self._bundle.permission_filter,
            audit_logger=self._bundle.audit_logger,
            permission_checker=self._bundle.permission_checker,
            permission_asker=self._bundle.permission_asker,
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
