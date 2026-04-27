"""FastAPI app 工厂：组装 provider、session store、路由与 lifespan。

lifespan 内统一初始化/销毁：
    - auth_config: 从 ServerConfig 构造；required 但无 token 时直接拒绝启动
    - provider (anthropic/openai): 按 config 单例
    - session_store: 按 provider + agent_factory 绑定
    - app.state.{provider, session_store, provider_name, config, auth_config}
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..agent.base import Agent
from ..agent.browser import BROWSER_SYSTEM_PROMPT
from ..agent.default import DEFAULT_SYSTEM_PROMPT, default_agent
from ..engine.sanitizer import DefaultSanitizer
from ..llm.provider import LLMProvider
from ..observability.logging import configure_json_logging
from ..database import DatabaseConfig, NullGateway, create_database
from ..ratelimit.config import RateLimitConfig
from ..ratelimit.limiter import RedisSlidingWindowLimiter
from ..ratelimit.middleware import RateLimitMiddleware
from ..ratelimit.redis_client import create_redis_client
from .auth import AuthConfig
from .chat import router as chat_router
from .config import ServerConfig
from .elicitation import HTTPElicitationBroker
from .elicitation_api import router as elicitation_router
from .images import router as images_router
from .plan import router as plan_router
from .sessions import SessionStore
from .sessions_api import router as sessions_router

if TYPE_CHECKING:
    from ..engine.permission.assignment import AssignmentStore
    from ..engine.permission.audit import AuditStore
    from ..engine.permission.killswitch import KillSwitchGate
    from ..engine.permission.persona_registry import PersonaRegistry
    from ..sandbox import OpenSandboxPool
    from .sessions import SessionEntry

_logger = logging.getLogger(__name__)


def _make_provider(provider_name: str, api_key: str, base_url: str | None) -> LLMProvider:
    if provider_name == "anthropic":
        mod = importlib.import_module("topsport_agent.llm.providers.anthropic")
        return mod.AnthropicProvider(api_key=api_key, base_url=base_url, max_tokens=4096)
    if provider_name == "openai":
        mod = importlib.import_module("topsport_agent.llm.providers.openai_chat")
        return mod.OpenAIChatProvider(api_key=api_key, base_url=base_url, max_tokens=4096)
    raise ValueError(f"unknown provider: {provider_name}")


def _build_auth_config(cfg: ServerConfig) -> AuthConfig:
    """按 ServerConfig 的鉴权字段构造 AuthConfig。优先级：tokens_file > 单 token > required 错误。"""
    if not cfg.auth_required:
        return AuthConfig.disabled()
    if cfg.auth_tokens_file:
        return AuthConfig.from_tokens_file(cfg.auth_tokens_file)
    if cfg.auth_token:
        return AuthConfig.from_single_token(cfg.auth_token)
    # required=True 但没给 token —— AuthConfig.__post_init__ 会抛，这里显式触发
    return AuthConfig(required=True, tokens={})


def _wrap_with_metrics(
    inner: Callable[[LLMProvider, str], Agent],
    metrics: Any,
) -> Callable[[LLMProvider, str], Agent]:
    """在 agent_factory 外层装饰：把 metrics subscriber 挂到新 agent。

    Agent.add_event_subscriber 内部同时更新 engine 与 capability bundle，
    确保 spawn_child 创建的 sub-agent 也被 metrics 采集（典型症状：plan
    mode 的 sub-step 不进 Langfuse / Prometheus）。"""

    def factory(provider: LLMProvider, model: str) -> Agent:
        agent = inner(provider, model)
        agent.add_event_subscriber(metrics)
        return agent

    return factory


def _wrap_with_extras(
    inner: Callable[[LLMProvider, str], Agent],
    tool_sources: list[Any],
    event_subscribers: list[Any],
) -> Callable[[LLMProvider, str], Agent]:
    """装饰 factory：把 MCP tool sources / Langfuse 等 event subscribers 挂到
    每个新 agent。与 _wrap_with_metrics 互不干扰，可叠加。

    走 Agent.add_tool_source / add_event_subscriber，这两个方法保证 engine
    runtime 与 capability bundle 同步更新；之前版本必须手工 mutate
    `_capability_bundle` dict 才不漏 sub-agent，重构后这条陷阱已关闭。
    """
    if not tool_sources and not event_subscribers:
        return inner

    def factory(provider: LLMProvider, model: str) -> Agent:
        agent = inner(provider, model)
        for src in tool_sources:
            agent.add_tool_source(src)
        for sub in event_subscribers:
            agent.add_event_subscriber(sub)
        return agent

    return factory


def _wrap_with_plan_checkpointer(
    inner: Callable[[LLMProvider, str], Agent],
    checkpointer: Any | None,
) -> Callable[[LLMProvider, str], Agent]:
    if checkpointer is None:
        return inner

    def factory(provider: LLMProvider, model: str) -> Agent:
        agent = inner(provider, model)
        agent.config.plan_checkpointer = checkpointer
        return agent

    return factory


def _build_mcp_manager(cfg: ServerConfig) -> Any | None:
    """Load MCP server manager from config file + built-in providers.

    Sources merged into a single MCPManager:
      1. JSON config file at MCP_CONFIG_PATH (claude-desktop format)
      2. Built-in Brave Search server when ENABLE_BRAVE_SEARCH=true

    Returns None when no source is configured. Fails fast on bad paths /
    missing API keys so operators see misconfigurations at startup rather
    than at first tool call.
    """
    if not cfg.mcp_config_path and not cfg.enable_brave_search:
        return None
    from ..mcp.builtin import brave_search_config
    from ..mcp.client import MCPClient
    from ..mcp.manager import MCPManager

    if cfg.mcp_config_path:
        manager = MCPManager.from_config_file(cfg.mcp_config_path)
    else:
        manager = MCPManager()

    if cfg.enable_brave_search:
        if not cfg.brave_api_key:
            raise RuntimeError(
                "ENABLE_BRAVE_SEARCH=true but BRAVE_API_KEY is empty"
            )
        brave_cfg = brave_search_config(cfg.brave_api_key)
        if manager.get(brave_cfg.name) is not None:
            raise RuntimeError(
                f"MCP server name {brave_cfg.name!r} already registered "
                f"via MCP_CONFIG_PATH; cannot also enable built-in Brave"
            )
        manager.register(MCPClient.from_config(brave_cfg))

    if cfg.mcp_roots:
        from ..mcp.roots import path_to_root, static_roots
        roots = [path_to_root(p) for p in cfg.mcp_roots]
        manager.set_roots_provider(static_roots(roots))

    if cfg.mcp_log_level and cfg.mcp_log_level not in {"off", "none"}:
        from typing import cast
        from ..mcp.logging_handler import (
            MCPLogLevel,
            default_logging_callback,
        )
        valid_levels = {
            "debug", "info", "notice", "warning",
            "error", "critical", "alert", "emergency",
        }
        if cfg.mcp_log_level not in valid_levels:
            raise RuntimeError(
                f"MCP_LOG_LEVEL must be one of {sorted(valid_levels)} / 'off' / 'none', "
                f"got {cfg.mcp_log_level!r}"
            )
        level = cast(MCPLogLevel, cfg.mcp_log_level)
        # 每个 client 用对应名字的 default_logging_callback，便于 ELK 按 server 过滤
        for client in manager.clients():
            client.set_logging_callback(
                default_logging_callback(client.name), level=level,
            )

    if cfg.enable_mcp_progress:
        from ..mcp.progress import default_progress_callback
        for client in manager.clients():
            client.set_progress_callback(default_progress_callback(client.name))

    # Elicitation handler is wired in lifespan after `app.state.elicitation_broker`
    # is created — broker can't exist before app instance does.
    return manager


def _build_langfuse_tracer(cfg: ServerConfig) -> Any | None:
    """Construct LangfuseTracer if enable_langfuse=True. Fail-fast when keys missing."""
    if not cfg.enable_langfuse:
        return None
    from ..observability.langfuse_tracer import LangfuseTracer

    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        raise RuntimeError(
            "ENABLE_LANGFUSE=true but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing"
        )
    return LangfuseTracer()


def _build_image_client(cfg: ServerConfig) -> Any | None:
    """Construct OpenAIImageGenerationClient if ENABLE_IMAGE_GEN=true.

    Fails fast if API_KEY missing or IMAGE_GEN_MODEL not set (operators expect
    a default model to hand to client requests that omit the `model` field).
    """
    if not cfg.enable_image_gen:
        return None
    from ..llm.image_generation import OpenAIImageGenerationClient

    if not cfg.api_key:
        raise RuntimeError("ENABLE_IMAGE_GEN=true but API_KEY missing")
    if not cfg.image_gen_model:
        raise RuntimeError(
            "ENABLE_IMAGE_GEN=true but IMAGE_GEN_MODEL is empty — set a default model"
        )
    image_base = cfg.image_gen_base_url or cfg.base_url

    def _factory() -> Any:
        openai_mod = importlib.import_module("openai")
        return openai_mod.AsyncOpenAI(api_key=cfg.api_key, base_url=image_base)

    return OpenAIImageGenerationClient(
        client_factory=_factory,
        default_model=cfg.image_gen_model,
    )


def _build_plan_checkpointer(cfg: ServerConfig) -> Any | None:
    if cfg.plan_checkpointer is not None:
        return cfg.plan_checkpointer
    if not cfg.plan_checkpoint_dir:
        return None
    from ..engine.checkpoint import FileCheckpointer

    return FileCheckpointer(cfg.plan_checkpoint_dir)


def _build_server_system_prompt(cfg: ServerConfig) -> str | None:
    """Decide the system prompt handed to default_agent based on enabled capabilities.

    Context: DEFAULT_SYSTEM_PROMPT documents file/skills/memory/plugin tools but
    says nothing about browser. BROWSER_SYSTEM_PROMPT documents browser workflow
    (@ref snapshot model) but drops the other capability docs. When both are
    enabled we need a merge strategy.

    Return None to fall back to default_agent's built-in prompt (DEFAULT_SYSTEM_PROMPT).

    TODO(niko): choose the merge strategy. Options to consider:

      (A) Return BROWSER_SYSTEM_PROMPT when cfg.enable_browser else None.
          -> Simple, but loses file/skills/memory guidance when browser is on.

      (B) Return DEFAULT_SYSTEM_PROMPT + "\n\n" + <browser-section extracted from
          BROWSER_SYSTEM_PROMPT> when cfg.enable_browser.
          -> Keeps all docs. Requires picking which section of
             BROWSER_SYSTEM_PROMPT to inline (toolset + interaction model + workflow?).

      (C) Build a dynamic prompt: only include capability sections that are
          actually enabled via cfg.enable_* flags (file tools / skills / memory /
          plugins / browser).
          -> Cleanest for multi-tenant but most code. Overkill if the model
             tolerates unused-capability hints.

    Trade-offs:
    - prompt length vs model context budget
    - does the LLM get confused by capability hints for tools it can't see?
    - do we want browser workflow discipline (snapshot-before-click) strictly
      enforced, or just hinted?
    """
    if not cfg.enable_browser:
        return None
    return DEFAULT_SYSTEM_PROMPT + "\n\n" + BROWSER_SYSTEM_PROMPT


def _default_agent_factory(
    cfg: ServerConfig,
    sandbox_pool: "OpenSandboxPool | None" = None,
) -> Callable[[LLMProvider, str], Agent]:
    """生产默认 Agent：按 ServerConfig 闸门决定 file_ops / skills / memory /
    plugins / sandbox。

    默认全关；对外暴露必须显式 opt-in（ENABLE_* env）。此前版本只尊重
    enable_file_tools，其它三个被 default_agent 硬编码为 True —— 在
    server 链路里实际 bypass 了 ServerConfig 的闸门。现已全部透传。

    当 sandbox_pool 存在（SANDBOX_ENABLED=true）：
      - 注入 OpenSandboxToolSource（sandbox_shell/read_file/write_file）
      - 强制关闭本地 file_ops（即便 ENABLE_FILE_TOOLS=true 也不生效）
        —— 避免 LLM 越过沙箱直接读写宿主文件系统（SEC-001）
    """
    # sandbox 启用时本地 file_ops 必须关，避免两条路径并存的逃逸风险
    file_ops_enabled = cfg.enable_file_tools and sandbox_pool is None
    # Prompt injection guard：按 config 决定是否给新 agent 配 sanitizer。
    sanitizer = DefaultSanitizer() if cfg.prompt_injection_guard else None

    def factory(provider: LLMProvider, model: str) -> Agent:
        extra_tool_sources: list[Any] = []
        if sandbox_pool is not None:
            # 延迟 import 避免未装 opensandbox 时拉起 module 报错
            from ..sandbox import OpenSandboxToolSource

            extra_tool_sources.append(OpenSandboxToolSource(sandbox_pool))

        return default_agent(
            provider=provider,
            model=model,
            stream=True,
            max_steps=cfg.max_chat_steps,
            system_prompt=_build_server_system_prompt(cfg),
            enable_browser=cfg.enable_browser,
            enable_file_ops=file_ops_enabled,
            enable_skills=cfg.enable_skills,
            enable_memory=cfg.enable_memory,
            enable_plugins=cfg.enable_plugins,
            extra_tool_sources=extra_tool_sources or None,
            sanitizer=sanitizer,
        )

    return factory


def _build_sandbox_pool(cfg: ServerConfig) -> "OpenSandboxPool | None":
    """按 cfg.sandbox_enabled 构造 OpenSandboxPool；失败则抛（阻塞启动）。"""
    if not cfg.sandbox_enabled:
        return None
    from ..sandbox import OpenSandboxPool

    return OpenSandboxPool.from_config(
        domain=cfg.sandbox_domain,
        image=cfg.sandbox_image,
        use_server_proxy=cfg.sandbox_use_server_proxy,
        per_tenant_max_sandboxes=cfg.sandbox_per_tenant_max,
        per_tenant_acquire_timeout=cfg.sandbox_per_tenant_timeout_seconds,
        idle_pause_seconds=cfg.sandbox_idle_pause_seconds,
    )


def create_app(
    config: ServerConfig | None = None,
    *,
    provider_name: str = "anthropic",
    provider: LLMProvider | None = None,
    agent_factory: Callable[[LLMProvider, str], Agent] | None = None,
    metrics: Any | None = None,
    sandbox_pool: "OpenSandboxPool | None" = None,
    persona_registry: "PersonaRegistry | None" = None,
    audit_store: "AuditStore | None" = None,
    kill_switch: "KillSwitchGate | None" = None,
    assignment_store: "AssignmentStore | None" = None,
) -> FastAPI:
    """构造一个绑定好依赖的 FastAPI app。

    provider / agent_factory 可由测试注入 mock；生产默认从 env 加载。
    metrics 是可选的 PrometheusMetrics 实例；传入后暴露 /metrics endpoint
    并把它注入到每个新 session 的 agent event_subscribers。
    sandbox_pool：测试注入用，跳过真实 OpenSandbox 调用。生产从 cfg.sandbox_enabled
    自动构造。
    """

    cfg = config or ServerConfig.from_env()
    # 结构化日志：进程级一次性配置，幂等可重入（重复 create_app 不重复挂 handler）。
    if cfg.log_format == "json":
        level = getattr(logging, cfg.log_level, logging.INFO)
        configure_json_logging(level=level)
    auth_config = _build_auth_config(cfg)

    # sandbox pool：测试注入优先；否则按 cfg.sandbox_enabled 构造
    pool = sandbox_pool if sandbox_pool is not None else _build_sandbox_pool(cfg)

    # 未显式注入 agent_factory 时，default factory 带上 sandbox tool_source
    raw_factory = agent_factory or _default_agent_factory(cfg, sandbox_pool=pool)

    # MCP / Langfuse / ImageGen：同步 opt-in，MCPManager.from_config_file 不做实际连接；
    # LangfuseTracer 构造时会 import langfuse（失败 fail-fast）；Image 客户端构造
    # 延迟到首次调用（client_factory 延迟 import openai）。
    mcp_manager = _build_mcp_manager(cfg)
    langfuse_tracer = _build_langfuse_tracer(cfg)
    image_client = _build_image_client(cfg)
    plan_checkpointer = _build_plan_checkpointer(cfg)

    extra_tool_sources: list[Any] = []
    if mcp_manager is not None:
        extra_tool_sources.extend(mcp_manager.tool_sources())
    extra_event_subs: list[Any] = []
    if langfuse_tracer is not None:
        extra_event_subs.append(langfuse_tracer)

    # 叠加装饰：metrics → extras(MCP/Langfuse)
    factory: Callable[[LLMProvider, str], Agent] = raw_factory
    factory = _wrap_with_plan_checkpointer(factory, plan_checkpointer)
    if metrics is not None:
        factory = _wrap_with_metrics(factory, metrics)
    factory = _wrap_with_extras(factory, extra_tool_sources, extra_event_subs)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        prov = provider
        if prov is None:
            if not cfg.api_key:
                raise RuntimeError(
                    "API_KEY not set — required to boot provider"
                )
            prov = _make_provider(provider_name, cfg.api_key, cfg.base_url)

        # MCP sampling handler 必须延迟到 prov 准备好之后挂载（_build_mcp_manager
        # 在 create_app 同步段执行时还没有 provider 实例）。fail-fast：
        # ENABLE_MCP_SAMPLING=true 但 default_model 为空时无法选模型 → 启动失败。
        # 无 mcp_manager 时打 warning（与 default_model 缺失的 fail-fast 不对称
        # —— enable_mcp_sampling + 无 server 是配置错位但非阻塞）。
        if cfg.enable_mcp_sampling:
            if mcp_manager is None:
                _logger.warning(
                    "ENABLE_MCP_SAMPLING=true but no MCP servers configured "
                    "(MCP_CONFIG_PATH unset and ENABLE_BRAVE_SEARCH=false); "
                    "sampling handler not attached"
                )
            else:
                from ..mcp.sampling import (
                    LLMProviderSamplingHandler, TokenBucketRateLimit,
                )
                if not cfg.default_model:
                    raise RuntimeError(
                        "ENABLE_MCP_SAMPLING=true but MODEL is empty — "
                        "sampling handler needs a default model to resolve "
                        "server requests when no hint matches"
                    )
                # 每个 client 一个独立 handler 实例，audit 日志自动携带 client name。
                # 速率限制按 client 隔离（每个 bucket 独立），避免单 server 刷穷
                # 整个 sampling quota；默认 60/min 是保守 baseline，operator 可
                # 通过 MCP_SAMPLING_RATE_LIMIT_PER_MIN 调整。
                rate_per_min = cfg.mcp_sampling_rate_limit_per_min
                for client in mcp_manager.clients():
                    rate = (
                        TokenBucketRateLimit(
                            capacity=rate_per_min,
                            refill_per_minute=rate_per_min,
                        )
                        if rate_per_min > 0 else None
                    )
                    client.set_sampling_handler(
                        LLMProviderSamplingHandler(
                            prov,
                            default_model=cfg.default_model,
                            max_tokens_cap=cfg.mcp_sampling_max_tokens,
                            rate_limit=rate,
                            client_name=client.name,
                        )
                    )

        # Elicitation broker — singleton routing server-driven user
        # prompts through the active chat SSE stream. Stays alive for
        # the app lifetime; in-memory state acceptable for v1 (single
        # replica). Multi-replica deployments need Redis-backed broker.
        broker: HTTPElicitationBroker | None = None
        if cfg.enable_mcp_elicitation and mcp_manager is not None:
            broker = HTTPElicitationBroker(
                default_timeout_seconds=cfg.mcp_elicitation_timeout_seconds,
            )
            handler = broker.make_handler()
            for client in mcp_manager.clients():
                client.set_elicitation_handler(handler)  # type: ignore[arg-type]
        app.state.elicitation_broker = broker

        # sandbox 生命周期挂到 SessionStore
        create_hooks: list = []
        close_hooks: list = []
        if pool is not None:
            from ..sandbox import SessionSandboxBinding

            binding = SessionSandboxBinding(pool)
            create_hooks.append(binding.on_session_created)
            close_hooks.append(binding.on_session_closed)

        # Capability-ACL control/execution plane bridge: if the operator wired
        # a persona_registry + assignment_store, resolve the session's persona
        # at creation time and copy the permissions into session.granted_permissions.
        # Without this hook, Assignment CRUD lives in isolation and never reaches
        # a running session — the exact bifurcation codex flagged.
        if persona_registry is not None and assignment_store is not None:
            persona_hook = _build_persona_resolver_hook(
                persona_registry, assignment_store,
            )
            create_hooks.append(persona_hook)

        # Per-session workspace: binds session.workspace.files_dir as the
        # file_ops sandbox root. Without this hook ToolContext.workspace_root
        # stays None and file tools run in unbounded CLI trust mode — a major
        # multi-tenant escape vector.
        from ..workspace import WorkspaceRegistry

        ws_base = Path(cfg.workspace_root) if cfg.workspace_root else (
            Path.home() / ".topsport-agent" / "workspaces"
        )
        ws_registry = WorkspaceRegistry(ws_base)

        async def _assign_workspace(session_id: str, entry: "SessionEntry") -> None:
            del session_id
            entry.session.workspace = ws_registry.acquire(entry.session.id)

        create_hooks.append(_assign_workspace)

        if cfg.workspace_delete_on_close:
            async def _release_workspace(session_id: str, entry: "SessionEntry") -> None:
                del entry
                ws_registry.release(session_id, delete=True)

            close_hooks.append(_release_workspace)

        app.state.workspace_registry = ws_registry

        store = SessionStore(
            agent_factory=factory,
            provider=prov,
            max_sessions=cfg.max_sessions,
            ttl_seconds=cfg.session_ttl_seconds,
            on_session_created=create_hooks or None,
            on_session_closed=close_hooks or None,
        )
        app.state.provider = prov
        app.state.provider_name = provider_name
        app.state.session_store = store
        app.state.agent_factory = factory
        app.state.config = cfg
        app.state.auth_config = auth_config
        app.state.sandbox_pool = pool
        app.state.image_client = image_client
        app.state.image_default_model = cfg.image_gen_model or None
        app.state.mcp_manager = mcp_manager
        app.state.langfuse_tracer = langfuse_tracer
        # === Database (optional, default off) ===
        if cfg.enable_database:
            if not cfg.database_url:
                raise RuntimeError(
                    "ENABLE_DATABASE=true but DATABASE_URL is unset"
                )
            db_config = DatabaseConfig(
                backend=cfg.database_backend,
                url=cfg.database_url,
                pool_min=cfg.database_pool_min,
                pool_max=cfg.database_pool_max,
                timeout_seconds=cfg.database_timeout_seconds,
            )
            db = create_database(db_config)
            await db.connect()
            # Assign to app.state BEFORE health_check so the finally-block
            # teardown can close the pool even if health_check fails
            # (otherwise a failed health_check leaks the asyncpg pool).
            app.state.database = db
            if not await db.health_check():
                raise RuntimeError(
                    "database enabled but health_check failed"
                )
        else:
            app.state.database = NullGateway()

        # === Rate limit (optional, default off) ===
        if cfg.enable_rate_limit:
            if not cfg.ratelimit_redis_url:
                raise RuntimeError(
                    "ENABLE_RATE_LIMIT=true but RATELIMIT_REDIS_URL is unset"
                )
            rl_client = create_redis_client(cfg.ratelimit_redis_url)
            try:
                if not await rl_client.ping():
                    raise RuntimeError(
                        "rate limit enabled but Redis ping failed"
                    )
            except Exception as exc:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(
                    f"rate limit enabled but Redis connection failed: {exc!r}"
                ) from exc
            lua_path = (
                Path(__file__).resolve().parent.parent
                / "ratelimit" / "lua" / "sliding_window.lua"
            )
            lua_script = lua_path.read_text(encoding="utf-8")
            sha = await rl_client.script_load(lua_script)
            rl_limiter = RedisSlidingWindowLimiter(
                client=rl_client, sha=sha, script=lua_script
            )
            app.state.ratelimit_client = rl_client
            app.state.ratelimit_limiter = rl_limiter
        else:
            app.state.ratelimit_limiter = None

        app.state.draining = False
        app.state.inflight = 0
        if not auth_config.required:
            _logger.warning(
                "server starting with auth DISABLED — acceptable only for CLI "
                "or isolated test environments; do not expose externally"
            )
        try:
            yield
        finally:
            # H-R5 graceful drain：先拒新请求，再等 in-flight 归零或超时，最后关资源
            app.state.draining = True
            deadline = time.monotonic() + cfg.drain_timeout_seconds
            while app.state.inflight > 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            if app.state.inflight > 0:
                _logger.warning(
                    "drain timed out with %d in-flight requests still active",
                    app.state.inflight,
                )
            await store.close_all()
            # Close MCP listening sessions (if any client subscribed).
            if mcp_manager is not None:
                try:
                    await mcp_manager.close_all()
                except Exception as exc:
                    _logger.warning(
                        "mcp manager close_all failed: %r", exc,
                    )
            if pool is not None:
                try:
                    await pool.close_all()
                except Exception as exc:
                    _logger.warning("sandbox pool close failed: %r", exc)
            # Close database if it was a real backend (NullGateway.close is no-op).
            db_state = getattr(app.state, "database", None)
            if db_state is not None:
                try:
                    await db_state.close()
                except Exception as exc:
                    _logger.warning("database close failed during shutdown: %r", exc)
            # Close rate-limit Redis client (if it was created).
            rl_client_state = getattr(app.state, "ratelimit_client", None)
            if rl_client_state is not None:
                try:
                    await rl_client_state.aclose()
                except Exception as exc:
                    _logger.warning(
                        "ratelimit client close failed: %r", exc
                    )

    app = FastAPI(
        title="topsport-agent",
        version="0.0.1",
        description="OpenAI-compatible chat + Plan execution API for topsport-agent",
        lifespan=lifespan,
    )

    class _DrainMiddleware(BaseHTTPMiddleware):
        """H-R5: 正在 drain 时，对 /v1/* 新请求返回 503；健康检查和 /metrics 放行。
        正常路径跟踪 in-flight 计数，lifespan 等它归零再继续关资源。
        """

        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            is_api = path.startswith("/v1/")
            if is_api and getattr(request.app.state, "draining", False):
                return JSONResponse(
                    status_code=503,
                    content={"error": "server draining, please retry"},
                    headers={"Retry-After": "5"},
                )
            if is_api:
                request.app.state.inflight = (
                    getattr(request.app.state, "inflight", 0) + 1
                )
                try:
                    return await call_next(request)
                finally:
                    request.app.state.inflight = max(
                        request.app.state.inflight - 1, 0
                    )
            return await call_next(request)

    app.add_middleware(_DrainMiddleware)

    # Rate limit middleware — installs only when enabled. Uses a lazy shim
    # that reads the real limiter from app.state at request time, so the
    # add_middleware call can happen before lifespan runs.
    if cfg.enable_rate_limit:
        rl_config = RateLimitConfig(
            enabled=True,
            redis_url=cfg.ratelimit_redis_url,
            window_seconds=cfg.ratelimit_window_seconds,
            per_ip_limit=cfg.ratelimit_per_ip,
            per_principal_limit=cfg.ratelimit_per_principal,
            per_tenant_limit=cfg.ratelimit_per_tenant,
            per_route_default=cfg.ratelimit_per_route_default,
            per_route_limits=dict(cfg.ratelimit_routes),
            trust_forwarded_for=cfg.ratelimit_trust_forwarded_for,
            fail_open_on_redis_error=cfg.ratelimit_fail_open,
        )

        class _LazyLimiter:
            """Defer to the live limiter stored on app.state at request time.

            The middleware is wired up at app creation, but the real limiter
            is only instantiated inside lifespan. Until lifespan runs, this
            shim returns allow-all (matches "not yet enforced" semantics).
            """

            async def check(self, rules):
                real = getattr(app.state, "ratelimit_limiter", None)
                if real is None:
                    from topsport_agent.ratelimit.types import (
                        RateLimitDecision,
                    )
                    return RateLimitDecision(
                        allowed=True,
                        denied_scope=None,
                        limit=0,
                        remaining=0,
                        reset_at_ms=0,
                        retry_after_seconds=0,
                    )
                return await real.check(rules)

        app.add_middleware(
            RateLimitMiddleware,
            limiter=_LazyLimiter(),
            config=rl_config,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """浅心跳：进程存活即 200，不依赖任何后端资源。"""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        """深检查：provider / session_store / auth_config 都已就绪才 200。
        任一缺失返回 503 —— LB 可据此把流量从损坏实例切走。
        """
        from fastapi import HTTPException

        components = {
            "provider": getattr(app.state, "provider", None) is not None,
            "session_store": getattr(app.state, "session_store", None) is not None,
            "auth_config": getattr(app.state, "auth_config", None) is not None,
        }
        if not all(components.values()):
            raise HTTPException(status_code=503, detail={"components": components})
        return {"status": "ready", "components": components}

    if metrics is not None:
        app.state.metrics = metrics

        @app.get("/metrics")
        async def metrics_endpoint() -> Response:
            payload, content_type = metrics.render()
            return Response(content=payload, media_type=content_type)

    app.include_router(chat_router)
    app.include_router(plan_router)
    app.include_router(sessions_router)
    app.include_router(images_router)
    app.include_router(elicitation_router)

    # Permission admin API 可选挂载：三件依赖都齐才暴露 /v1/admin/* 路由，
    # 缺任一（典型 CLI/测试场景）保持不变。
    # assignment_store 可选：缺它时 /assignments 路由也挂，但调用返回 501。
    if persona_registry is not None and audit_store is not None and kill_switch is not None:
        from .permission_api import build_permission_router

        app.include_router(
            build_permission_router(
                persona_registry=persona_registry,
                audit_store=audit_store,
                kill_switch=kill_switch,
                assignment_store=assignment_store,
            ),
            prefix="/v1/admin",
        )
    return app


def _build_persona_resolver_hook(
    persona_registry: "PersonaRegistry",
    assignment_store: "AssignmentStore",
) -> Callable[[str, "SessionEntry"], Any]:
    """Construct a SessionStore create-hook that populates
    `session.granted_permissions` from the first applicable PersonaAssignment.

    Resolution order (matches engine.permission.assignment.resolve_persona_ids):
        1. (tenant_id, user_id) — per-user override
        2. (tenant_id, group_id) — group default (not used by server layer today
           because user→group mapping is out of scope for v1; kept for future)
        3. (tenant_id, None, None) — tenant-wide default

    When no assignment matches, leave granted_permissions=frozenset() — the
    ToolVisibilityFilter will filter every tagged tool. This is fail-closed:
    secure-by-default for enterprise deployments.
    """
    from ..engine.permission.assignment import resolve_persona_ids

    async def hook(session_id: str, entry: "SessionEntry") -> None:
        session = entry.session
        # server.chat.py maps tenant_id = principal (simplest case). If the
        # mapping changes, the resolver still just uses whatever tenant_id
        # the session carries.
        tenant = session.tenant_id
        if not tenant:
            return
        user_id = session.principal
        resolved = await resolve_persona_ids(
            assignment_store, tenant_id=tenant, user_id=user_id,
        )
        if resolved is None:
            return
        _persona_ids, default_persona_id = resolved
        if default_persona_id is None:
            return
        persona = await persona_registry.get(default_persona_id)
        if persona is None:
            return
        session.granted_permissions = persona.permissions
        session.persona_id = persona.id

    return hook


__all__ = ["create_app"]
