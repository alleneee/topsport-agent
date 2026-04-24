"""RateLimiter Protocol and its Redis implementation."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .types import RateLimitDecision, RateLimitRule

_KEY_PREFIX = "ratelimit:sw:v1"


@runtime_checkable
class RateLimiter(Protocol):
    """Atomic multi-dimension rate-limit check.

    Contract:
    - If ANY rule in `rules` is over its quota, deny the request and do NOT
      increment any rule's counter.
    - If ALL rules pass, record the request in every dimension atomically.
    - Never partially-commits — failure mode must preserve all-or-nothing.
    """

    async def check(
        self, rules: Sequence[RateLimitRule]
    ) -> RateLimitDecision: ...


class RedisSlidingWindowLimiter:
    """Sliding-window rate limiter backed by Redis + a Lua script.

    One `SCRIPT LOAD` at startup gives a sha1; per request we try `EVALSHA`
    and fall back to `EVAL` on NOSCRIPT (auto-heals after Redis restart or
    SCRIPT FLUSH).
    """

    def __init__(
        self,
        *,
        client: Any,
        sha: str,
        script: str,
    ) -> None:
        self._client = client
        self._sha = sha
        self._script = script

    @staticmethod
    def _key(rule: RateLimitRule) -> str:
        return f"{_KEY_PREFIX}:{rule.scope.value}:{rule.identity}"

    async def check(
        self, rules: Sequence[RateLimitRule]
    ) -> RateLimitDecision:
        if not rules:
            return RateLimitDecision(
                allowed=True,
                denied_scope=None,
                limit=0,
                remaining=0,
                reset_at_ms=0,
                retry_after_seconds=0,
            )

        now_ms = int(time.time() * 1000)
        suffix = uuid.uuid4().hex[:12]

        keys = [self._key(r) for r in rules]
        args: list[int | str] = [now_ms, suffix]
        for rule in rules:
            args.append(rule.limit)
            args.append(rule.window_seconds * 1000)

        try:
            raw = await self._client.evalsha(
                self._sha, len(keys), *keys, *args
            )
        except Exception as exc:  # noqa: BLE001
            exc_repr = repr(exc)
            if "NOSCRIPT" in exc_repr or "NoScript" in exc_repr or type(exc).__name__ == "NoScriptError":
                raw = await self._client.eval(
                    self._script, len(keys), *keys, *args
                )
            else:
                raise

        allowed_flag, denied_idx, count, limit, reset_at_ms = (
            int(x) for x in raw
        )
        if allowed_flag == 1:
            primary = rules[0]
            return RateLimitDecision(
                allowed=True,
                denied_scope=None,
                limit=primary.limit,
                remaining=max(primary.limit - 1, 0),
                reset_at_ms=now_ms + primary.window_seconds * 1000,
                retry_after_seconds=0,
            )

        denied_rule = rules[denied_idx - 1]  # Lua indices 1-based
        retry_after = max(1, (reset_at_ms - now_ms + 999) // 1000)
        return RateLimitDecision(
            allowed=False,
            denied_scope=denied_rule.scope,
            limit=limit,
            remaining=0,
            reset_at_ms=reset_at_ms,
            retry_after_seconds=int(retry_after),
        )


__all__ = ["RateLimiter", "RedisSlidingWindowLimiter"]
