from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_redis_available = importlib.util.find_spec("redis") is not None
_test_url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
requires_redis = pytest.mark.skipif(
    not _redis_available, reason="redis package not installed"
)


@pytest.fixture
async def app_and_client():
    """Build a minimal FastAPI app with the RateLimitMiddleware wired up."""
    from httpx import ASGITransport, AsyncClient

    try:
        from fastapi import FastAPI
    except ImportError:
        pytest.skip("fastapi not installed")

    from topsport_agent.ratelimit.config import RateLimitConfig
    from topsport_agent.ratelimit.limiter import RedisSlidingWindowLimiter
    from topsport_agent.ratelimit.middleware import RateLimitMiddleware
    from topsport_agent.ratelimit.redis_client import create_redis_client

    client = create_redis_client(_test_url)
    try:
        if not await client.ping():
            pytest.skip("local Redis not reachable")
    except Exception:
        pytest.skip("local Redis not reachable")
    await client.flushdb()

    script = Path(
        "src/topsport_agent/ratelimit/lua/sliding_window.lua"
    ).resolve().read_text(encoding="utf-8")
    sha = await client.script_load(script)
    limiter = RedisSlidingWindowLimiter(
        client=client, sha=sha, script=script
    )

    cfg = RateLimitConfig(
        enabled=True,
        redis_url=_test_url,
        per_ip_limit=3,
        per_principal_limit=0,
        per_tenant_limit=0,
        window_seconds=60,
    )

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limiter=limiter, config=cfg)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"ok": "yes"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield app, ac, client

    await client.flushdb()
    await client.aclose()


@requires_redis
@pytest.mark.asyncio
async def test_nth_plus_one_returns_429(app_and_client) -> None:
    _, ac, _ = app_and_client
    for _ in range(3):
        r = await ac.get("/ping")
        assert r.status_code == 200
    r = await ac.get("/ping")
    assert r.status_code == 429
    body = r.json()
    assert body["error"] == "rate_limited"
    assert body["scope"] == "ip"
    assert "retry_after" in body
    assert "Retry-After" in r.headers
    assert "X-RateLimit-Scope" in r.headers
    assert r.headers["X-RateLimit-Scope"] == "ip"


@requires_redis
@pytest.mark.asyncio
async def test_exempt_path_bypasses_limit(app_and_client) -> None:
    _, ac, _ = app_and_client
    for _ in range(10):
        r = await ac.get("/health")
        assert r.status_code == 200


@requires_redis
@pytest.mark.asyncio
async def test_successful_response_has_ratelimit_headers(app_and_client) -> None:
    _, ac, _ = app_and_client
    r = await ac.get("/ping")
    assert r.status_code == 200
    assert "X-RateLimit-Limit" in r.headers
    assert "X-RateLimit-Remaining" in r.headers


@requires_redis
@pytest.mark.asyncio
async def test_fail_open_when_redis_disconnects(app_and_client) -> None:
    """Close the Redis client underneath the limiter; request still succeeds."""
    _, ac, client = app_and_client

    await client.aclose()
    # limiter now has a closed client; next check() will raise.

    r = await ac.get("/ping")
    # Default fail_open_on_redis_error=True → request served.
    assert r.status_code == 200
