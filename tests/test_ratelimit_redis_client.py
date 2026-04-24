import importlib.util
import os

import pytest

from topsport_agent.ratelimit.redis_client import create_redis_client

_redis_available = importlib.util.find_spec("redis") is not None
_test_url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.mark.skipif(not _redis_available, reason="redis package not installed")
def test_create_redis_client_returns_async_client() -> None:
    client = create_redis_client(_test_url)
    for name in ("ping", "evalsha", "eval", "script_load", "close", "flushdb"):
        assert hasattr(client, name), f"missing method: {name}"


@pytest.mark.skipif(not _redis_available, reason="redis package not installed")
@pytest.mark.asyncio
async def test_ping_round_trip_if_redis_up() -> None:
    client = create_redis_client(_test_url)
    try:
        ok = await client.ping()
    except Exception:
        pytest.skip("local Redis not reachable at " + _test_url)
    assert ok is True
    await client.close()


def test_create_without_redis_package_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `redis` is not installed, factory must raise ImportError with an install hint."""
    import sys

    monkeypatch.setitem(sys.modules, "redis", None)
    monkeypatch.setitem(sys.modules, "redis.asyncio", None)

    with pytest.raises(ImportError, match="uv sync --group redis"):
        create_redis_client(_test_url)
