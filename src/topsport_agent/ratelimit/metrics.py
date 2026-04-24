"""Prometheus metrics for the rate limiter. No-op when prometheus_client missing."""

from __future__ import annotations

import importlib
from typing import Any


class _NoOpCounter:
    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpCounter":
        return self

    def inc(self, amount: float = 1.0) -> None:
        return None


class _NoOpHistogram:
    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpHistogram":
        return self

    def observe(self, value: float) -> None:
        return None


def _try_load_prometheus() -> Any | None:
    try:
        return importlib.import_module("prometheus_client")
    except ImportError:
        return None


class RateLimitMetrics:
    """Three counters + one histogram. All become no-ops when prometheus_client
    isn't installed, matching the project's optional-dep style.
    """

    def __init__(self, *, registry: Any | None = None) -> None:
        prom = _try_load_prometheus()
        if prom is None:
            self._requests = _NoOpCounter()
            self._denied = _NoOpCounter()
            self._degraded = _NoOpCounter()
            self._duration = _NoOpHistogram()
            return

        kwargs = {"registry": registry} if registry is not None else {}
        self._requests = prom.Counter(
            "ratelimit_requests_total",
            "Rate-limit checks by scope",
            ["scope"],
            **kwargs,
        )
        self._denied = prom.Counter(
            "ratelimit_denied_total",
            "Rate-limit denials by scope",
            ["scope"],
            **kwargs,
        )
        self._degraded = prom.Counter(
            "ratelimit_degraded_total",
            "Rate-limit checks that fell back to fail-open due to Redis error",
            ["reason"],
            **kwargs,
        )
        self._duration = prom.Histogram(
            "ratelimit_check_duration_seconds",
            "Rate-limit EVAL/EVALSHA duration",
            **kwargs,
        )

    def inc_request(self, scope: str) -> None:
        self._requests.labels(scope=scope).inc()

    def inc_denied(self, scope: str) -> None:
        self._denied.labels(scope=scope).inc()

    def inc_degraded(self, reason: str) -> None:
        self._degraded.labels(reason=reason).inc()

    def observe_check_duration(self, seconds: float) -> None:
        self._duration.observe(seconds)


__all__ = ["RateLimitMetrics"]
