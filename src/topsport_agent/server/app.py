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
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..agent.base import Agent
from ..agent.default import default_agent
from ..engine.sanitizer import DefaultSanitizer
from ..llm.provider import LLMProvider
from ..observability.logging import configure_json_logging
from ..database import DatabaseConfig, NullGateway, create_database
from .auth import AuthConfig
from .chat import router as chat_router
from .config import ServerConfig
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
    """在 agent_factory 外层装饰：把 metrics subscriber 挂到新 agent 的 engine。"""

    def factory(provider: LLMProvider, model: str) -> Agent:
        agent = inner(provider, model)
        agent.engine.add_event_subscriber(metrics)
        return agent

    return factory


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
            enable_browser=False,
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

    # 若传入 metrics，把它装到每个新 session 的 agent 事件订阅里
    if metrics is not None:
        factory = _wrap_with_metrics(raw_factory, metrics)
    else:
        factory = raw_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        prov = provider
        if prov is None:
            if not cfg.api_key:
                raise RuntimeError(
                    "API_KEY not set — required to boot provider"
                )
            prov = _make_provider(provider_name, cfg.api_key, cfg.base_url)

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
