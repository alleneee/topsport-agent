"""FastAPI app 工厂：组装 provider、session store、路由与 lifespan。

lifespan 内统一初始化/销毁：
    - auth_config: 从 ServerConfig 构造；required 但无 token 时直接拒绝启动
    - provider (anthropic/openai): 按 config 单例
    - session_store: 按 provider + agent_factory 绑定
    - app.state.{provider, session_store, provider_name, config, auth_config}
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..agent.base import Agent
from ..agent.default import default_agent
from ..llm.provider import LLMProvider
from .auth import AuthConfig
from .chat import router as chat_router
from .config import ServerConfig
from .plan import router as plan_router
from .sessions import SessionStore

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


def _default_agent_factory(cfg: ServerConfig) -> Callable[[LLMProvider, str], Agent]:
    """生产默认 Agent：按 ServerConfig 闸门决定是否开 file_ops / skills / plugins。

    默认关文件 / 技能 / 插件 —— 对外暴露必须显式 opt-in（ENABLE_* env）。
    开 file_tools 时会把 workspace_root 注入到 agent 所有工具调用（由 session-level 实现）。
    """

    def factory(provider: LLMProvider, model: str) -> Agent:
        return default_agent(
            provider=provider,
            model=model,
            stream=True,
            enable_browser=False,
            enable_file_ops=cfg.enable_file_tools,
        )

    return factory


def create_app(
    config: ServerConfig | None = None,
    *,
    provider_name: str = "anthropic",
    provider: LLMProvider | None = None,
    agent_factory: Callable[[LLMProvider, str], Agent] | None = None,
) -> FastAPI:
    """构造一个绑定好依赖的 FastAPI app。

    provider / agent_factory 可由测试注入 mock；生产默认从 env 加载。
    """

    cfg = config or ServerConfig.from_env()
    factory = agent_factory or _default_agent_factory(cfg)
    auth_config = _build_auth_config(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        prov = provider
        if prov is None:
            if not cfg.api_key:
                raise RuntimeError(
                    "API_KEY not set — required to boot provider"
                )
            prov = _make_provider(provider_name, cfg.api_key, cfg.base_url)

        store = SessionStore(
            agent_factory=factory,
            provider=prov,
            max_sessions=cfg.max_sessions,
            ttl_seconds=cfg.session_ttl_seconds,
        )
        app.state.provider = prov
        app.state.provider_name = provider_name
        app.state.session_store = store
        app.state.config = cfg
        app.state.auth_config = auth_config
        if not auth_config.required:
            _logger.warning(
                "server starting with auth DISABLED — acceptable only for CLI "
                "or isolated test environments; do not expose externally"
            )
        try:
            yield
        finally:
            await store.close_all()

    app = FastAPI(
        title="topsport-agent",
        version="0.0.1",
        description="OpenAI-compatible chat + Plan execution API for topsport-agent",
        lifespan=lifespan,
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

    app.include_router(chat_router)
    app.include_router(plan_router)
    return app


__all__ = ["create_app"]
