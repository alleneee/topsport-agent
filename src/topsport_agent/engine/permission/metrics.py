"""PermissionMetrics — EventSubscriber that counts per-tool outcomes.

Plugs into Engine via event_subscribers. Exposes an in-memory counter
snapshot consumable by Prometheus exporters in the deployment layer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ...types.events import Event, EventType

__all__ = ["PermissionMetrics"]


class PermissionMetrics:
    name = "permission_metrics"

    def __init__(self) -> None:
        self._tool_calls: Counter[tuple[str, str]] = Counter()

    async def on_event(self, event: Event) -> None:
        if event.type != EventType.TOOL_CALL_END:
            return
        name = str(event.payload.get("name", "?"))
        is_error = bool(event.payload.get("is_error", False))
        outcome = "error" if is_error else "allowed"
        self._tool_calls[(name, outcome)] += 1

    def snapshot(self) -> dict[str, Any]:
        return {"tool_calls": dict(self._tool_calls)}
