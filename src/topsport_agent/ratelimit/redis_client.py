"""Factory for the async Redis client.

Imports the `redis` package lazily via importlib so the module is safe to
import in environments without `--group redis`. The ImportError wrapper
surfaces an install-hint message at construction time.
"""

from __future__ import annotations

import importlib
from typing import Any


def create_redis_client(url: str) -> Any:
    """Construct an async Redis client from a redis:// URL.

    Returns an instance of `redis.asyncio.Redis` with `decode_responses=True`
    so EVALSHA returns Python strings and ZADD/ZCARD round-trip cleanly.

    Raises:
        ImportError: if the `redis` package is not installed.
    """
    try:
        mod_name = "redis.asyncio"
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ImportError(
            "RateLimiter requires the 'redis' package. Install via: "
            "uv sync --group redis"
        ) from exc
    return mod.Redis.from_url(url, decode_responses=True)


__all__ = ["create_redis_client"]
