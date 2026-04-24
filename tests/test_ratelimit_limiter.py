from __future__ import annotations

import asyncio
import importlib.util
import os
import uuid
from pathlib import Path

import pytest

from topsport_agent.ratelimit.redis_client import create_redis_client
from topsport_agent.ratelimit.types import RateLimitRule, RateLimitScope

_redis_available = importlib.util.find_spec("redis") is not None
_test_url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
requires_redis = pytest.mark.skipif(
    not _redis_available, reason="redis package not installed"
)


@pytest.fixture
async def redis_client():
    """Shared fixture: real Redis on DB 15, FLUSHDB before+after."""
    if not _redis_available:
        pytest.skip("redis package not installed")
    client = create_redis_client(_test_url)
    try:
        if not await client.ping():
            pytest.skip("local Redis not responding")
    except Exception:
        pytest.skip(f"local Redis not reachable at {_test_url}")
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest.fixture
async def limiter(redis_client):
    from topsport_agent.ratelimit.limiter import RedisSlidingWindowLimiter

    script_path = (
        Path("src/topsport_agent/ratelimit/lua/sliding_window.lua")
        .resolve()
    )
    script = script_path.read_text(encoding="utf-8")
    sha = await redis_client.script_load(script)
    return RedisSlidingWindowLimiter(
        client=redis_client,
        sha=sha,
        script=script,
    )


def _unique_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@requires_redis
@pytest.mark.asyncio
async def test_first_request_is_allowed(limiter) -> None:
    decision = await limiter.check(
        [
            RateLimitRule(
                scope=RateLimitScope.IP,
                identity=_unique_id("ip"),
                limit=5,
                window_seconds=60,
            )
        ]
    )
    assert decision.allowed is True
    assert decision.denied_scope is None


@requires_redis
@pytest.mark.asyncio
async def test_nth_plus_one_request_denied(limiter) -> None:
    ident = _unique_id("ip")
    rule = RateLimitRule(
        scope=RateLimitScope.IP,
        identity=ident,
        limit=3,
        window_seconds=60,
    )
    for _ in range(3):
        assert (await limiter.check([rule])).allowed is True
    decision = await limiter.check([rule])
    assert decision.allowed is False
    assert decision.denied_scope == RateLimitScope.IP
    assert decision.limit == 3
    assert decision.retry_after_seconds > 0


@requires_redis
@pytest.mark.asyncio
async def test_multi_dimension_denies_on_tightest(limiter) -> None:
    ip_id = _unique_id("ip")
    tenant_id = _unique_id("tenant")
    ip_rule = RateLimitRule(RateLimitScope.IP, ip_id, 100, 60)
    tenant_rule = RateLimitRule(RateLimitScope.TENANT, tenant_id, 2, 60)

    for _ in range(2):
        assert (await limiter.check([ip_rule, tenant_rule])).allowed is True

    decision = await limiter.check([ip_rule, tenant_rule])
    assert decision.allowed is False
    assert decision.denied_scope == RateLimitScope.TENANT


@requires_redis
@pytest.mark.asyncio
async def test_atomicity_no_partial_commit(limiter, redis_client) -> None:
    """When one rule denies, the OTHER rule's counter must not increment."""
    ip_id = _unique_id("ip")
    tenant_id = _unique_id("tenant")
    ip_rule = RateLimitRule(RateLimitScope.IP, ip_id, 100, 60)
    tenant_rule = RateLimitRule(RateLimitScope.TENANT, tenant_id, 1, 60)

    await limiter.check([ip_rule, tenant_rule])

    ip_key = f"ratelimit:sw:v1:ip:{ip_id}"
    tenant_key = f"ratelimit:sw:v1:tenant:{tenant_id}"
    ip_count_before = await redis_client.zcard(ip_key)
    tenant_count_before = await redis_client.zcard(tenant_key)
    assert ip_count_before == 1
    assert tenant_count_before == 1

    decision = await limiter.check([ip_rule, tenant_rule])
    assert decision.allowed is False

    ip_count_after = await redis_client.zcard(ip_key)
    tenant_count_after = await redis_client.zcard(tenant_key)
    assert ip_count_after == 1, "IP rule must not increment when tenant denies"
    assert tenant_count_after == 1, "tenant rule must not double-count"


@requires_redis
@pytest.mark.asyncio
async def test_noscript_fallback_triggers_eval(limiter, redis_client) -> None:
    """If EVALSHA gets NOSCRIPT, the limiter falls back to EVAL transparently."""
    await redis_client.script_flush()
    rule = RateLimitRule(
        scope=RateLimitScope.IP,
        identity=_unique_id("ip"),
        limit=5,
        window_seconds=60,
    )
    decision = await limiter.check([rule])
    assert decision.allowed is True


@requires_redis
@pytest.mark.asyncio
async def test_concurrent_requests_respect_quota(limiter) -> None:
    ident = _unique_id("ip")
    rule = RateLimitRule(RateLimitScope.IP, ident, 3, 60)
    results = await asyncio.gather(
        *[limiter.check([rule]) for _ in range(10)]
    )
    allowed = sum(1 for r in results if r.allowed)
    denied = sum(1 for r in results if not r.allowed)
    assert allowed == 3
    assert denied == 7
