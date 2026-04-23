"""Per-tenant emergency kill switch.

When a tenant is in the killed set, ToolVisibilityFilter returns an empty
tool pool — the LLM cannot invoke any tool on the next step. In-flight tool
calls complete normally; cancel_event is a separate concern.

In-memory implementation is the default. For multi-instance deployments,
replace with a Redis-backed implementation that implements the same public
methods (is_active / activate / deactivate / active_tenants).
"""

from __future__ import annotations

import threading


class KillSwitchGate:
    """Thread-safe in-memory kill switch."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._killed: set[str] = set()

    def is_active(self, tenant_id: str | None) -> bool:
        if tenant_id is None:
            return False
        with self._lock:
            return tenant_id in self._killed

    def activate(self, tenant_id: str) -> None:
        with self._lock:
            self._killed.add(tenant_id)

    def deactivate(self, tenant_id: str) -> None:
        with self._lock:
            self._killed.discard(tenant_id)

    def active_tenants(self) -> frozenset[str]:
        """Immutable snapshot of currently killed tenants."""
        with self._lock:
            return frozenset(self._killed)
