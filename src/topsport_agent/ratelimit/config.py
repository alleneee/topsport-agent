"""RateLimitConfig — all tunables for the rate-limit subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field

_DEFAULT_EXEMPT: frozenset[str] = frozenset(
    {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}
)


@dataclass(slots=True, frozen=True)
class RateLimitConfig:
    """Frozen config. Default is "off" and imposes zero runtime cost."""

    enabled: bool = False
    redis_url: str | None = None

    # Per-dimension limits. 0 = disable that dimension.
    window_seconds: int = 60
    per_ip_limit: int = 300
    per_principal_limit: int = 60
    per_tenant_limit: int = 1000
    per_route_default: int = 0
    per_route_limits: dict[str, int] = field(default_factory=dict)

    exempt_paths: frozenset[str] = _DEFAULT_EXEMPT
    trust_forwarded_for: bool = False
    fail_open_on_redis_error: bool = True


__all__ = ["RateLimitConfig"]
