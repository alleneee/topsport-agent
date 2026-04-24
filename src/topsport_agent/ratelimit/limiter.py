"""RateLimiter Protocol and its Redis implementation.

Protocol here; the RedisSlidingWindowLimiter class lands in a follow-up task.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import RateLimitDecision, RateLimitRule


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


__all__ = ["RateLimiter"]
