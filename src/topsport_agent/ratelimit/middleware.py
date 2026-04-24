"""RateLimitMiddleware — Starlette/ASGI rate-limit integration.

Runs AFTER auth middleware (so request.state.principal / tenant_id are set).
Skips exempt paths, extracts identity per scope, calls the limiter, and
maps decisions to 429 responses or X-RateLimit-* headers on success.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .config import RateLimitConfig
from .limiter import RateLimiter
from .metrics import RateLimitMetrics
from .types import RateLimitRule, RateLimitScope

_logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        limiter: RateLimiter,
        config: RateLimitConfig,
        metrics: RateLimitMetrics | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._config = config
        self._metrics = metrics or RateLimitMetrics()

    async def dispatch(self, request: Request, call_next):
        if self._is_exempt(request.url.path):
            return await call_next(request)

        rules = self._build_rules(request)
        if not rules:
            # All dimensions disabled — pass through.
            return await call_next(request)

        try:
            decision = await self._limiter.check(rules)
        except Exception as exc:  # noqa: BLE001
            self._metrics.inc_degraded(type(exc).__name__)
            _logger.warning(
                "ratelimit check failed; %s: %r",
                "allowing request (fail_open)"
                if self._config.fail_open_on_redis_error
                else "rejecting request",
                exc,
            )
            if self._config.fail_open_on_redis_error:
                return await call_next(request)
            return JSONResponse(
                status_code=503,
                content={"error": "ratelimit_unavailable"},
            )

        if not decision.allowed:
            scope = decision.denied_scope
            assert scope is not None  # invariant when allowed=False
            self._metrics.inc_denied(scope.value)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "scope": scope.value,
                    "retry_after": decision.retry_after_seconds,
                },
                headers={
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(decision.reset_at_ms // 1000),
                    "X-RateLimit-Scope": scope.value,
                },
            )

        # Allowed — attach informational headers to the downstream response.
        self._metrics.inc_request("any")
        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        return response

    # ---------- helpers ----------

    def _is_exempt(self, path: str) -> bool:
        return path in self._config.exempt_paths

    def _client_ip(self, request: Request) -> str:
        if self._config.trust_forwarded_for:
            xff = request.headers.get("x-forwarded-for")
            if xff:
                return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _route_template(self, request: Request) -> str:
        route = request.scope.get("route")
        # Starlette stores the original path template on the route object.
        path = getattr(route, "path", request.url.path)
        method = request.method
        return f"{method}:{path}"

    def _build_rules(self, request: Request) -> Sequence[RateLimitRule]:
        rules: list[RateLimitRule] = []
        cfg = self._config

        if cfg.per_ip_limit > 0:
            rules.append(
                RateLimitRule(
                    scope=RateLimitScope.IP,
                    identity=self._client_ip(request),
                    limit=cfg.per_ip_limit,
                    window_seconds=cfg.window_seconds,
                )
            )

        principal = getattr(request.state, "principal", None) or "anon"
        if cfg.per_principal_limit > 0:
            rules.append(
                RateLimitRule(
                    scope=RateLimitScope.PRINCIPAL,
                    identity=principal,
                    limit=cfg.per_principal_limit,
                    window_seconds=cfg.window_seconds,
                )
            )

        tenant = getattr(request.state, "tenant_id", None) or "public"
        if cfg.per_tenant_limit > 0:
            rules.append(
                RateLimitRule(
                    scope=RateLimitScope.TENANT,
                    identity=tenant,
                    limit=cfg.per_tenant_limit,
                    window_seconds=cfg.window_seconds,
                )
            )

        # Route-specific quotas
        route_key = self._route_template(request)
        route_limit = cfg.per_route_limits.get(route_key, cfg.per_route_default)
        if route_limit > 0:
            rules.append(
                RateLimitRule(
                    scope=RateLimitScope.ROUTE,
                    identity=route_key,
                    limit=route_limit,
                    window_seconds=cfg.window_seconds,
                )
            )

        return rules


__all__ = ["RateLimitMiddleware"]
