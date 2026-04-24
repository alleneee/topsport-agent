"""Lifespan wiring: enable_rate_limit=True + bad Redis → fail-fast at startup."""
from __future__ import annotations

import pytest

from topsport_agent.server.app import create_app
from topsport_agent.server.config import ServerConfig


def _stub_provider():
    class _P:
        name = "stub"

        async def complete(self, request):  # pragma: no cover
            raise RuntimeError("not used")

    return _P()


@pytest.mark.asyncio
async def test_disabled_ratelimit_does_not_touch_redis() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_rate_limit=False,
    )
    app = create_app(cfg, provider=_stub_provider())
    async with app.router.lifespan_context(app):
        assert getattr(app.state, "ratelimit_limiter", None) is None


@pytest.mark.asyncio
async def test_enabled_without_redis_url_fails_fast() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_rate_limit=True,
        ratelimit_redis_url=None,
    )
    app = create_app(cfg, provider=_stub_provider())
    with pytest.raises(RuntimeError, match="RATELIMIT_REDIS_URL"):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.asyncio
async def test_enabled_with_unreachable_redis_fails_fast() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_rate_limit=True,
        ratelimit_redis_url="redis://nonexistent.invalid:6379/0",
    )
    app = create_app(cfg, provider=_stub_provider())
    with pytest.raises(RuntimeError, match="Redis"):
        async with app.router.lifespan_context(app):
            pass
