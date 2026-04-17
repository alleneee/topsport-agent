"""FastAPI app 工厂：组装 provider、session store、路由与 lifespan。

lifespan 内统一初始化/销毁：
    - provider (anthropic/openai): 按 config 单例
    - session_store: 按 provider + agent_factory 绑定
    - app.state.{provider, session_store, provider_name} 供路由读取
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


def _default_agent_factory() -> Callable[[LLMProvider, str], Agent]:
    """每个 session 独立构造 Agent。默认开流式，关掉 browser 减少启动开销。"""

    def factory(provider: LLMProvider, model: str) -> Agent:
        return default_agent(
            provider=provider,
            model=model,
            stream=True,
            enable_browser=False,
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
    factory = agent_factory or _default_agent_factory()

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
        return {"status": "ok"}

    app.include_router(chat_router)
    app.include_router(plan_router)
    return app
