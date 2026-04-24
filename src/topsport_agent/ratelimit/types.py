"""Rate-limiter value types: scope enum, rule, decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RateLimitScope(str, Enum):
    """The four dimensions we enforce. Each has its own Redis key space."""

    IP = "ip"
    PRINCIPAL = "principal"
    TENANT = "tenant"
    ROUTE = "route"


@dataclass(slots=True, frozen=True)
class RateLimitRule:
    """One dimension's quota for one identity.

    The limiter checks N rules per request (typically 4, one per scope).
    scope + identity together form the Redis key namespace.
    """

    scope: RateLimitScope
    identity: str               # e.g. "1.2.3.4", "user-42", "acme", "POST:/chat"
    limit: int                  # requests allowed in window
    window_seconds: int         # window size in seconds


@dataclass(slots=True, frozen=True)
class RateLimitDecision:
    """Outcome of a `limiter.check(rules)` call.

    - allowed=True  → all rules within quota; remaining/limit are the tightest rule's metrics (for response headers)
    - allowed=False → denied_scope is the first rule that exceeded its quota
    """

    allowed: bool
    denied_scope: RateLimitScope | None
    limit: int
    remaining: int
    reset_at_ms: int
    retry_after_seconds: int


__all__ = ["RateLimitDecision", "RateLimitRule", "RateLimitScope"]
