from __future__ import annotations

from ..engine.hooks import EventSubscriber
from ..types.events import Event

Tracer = EventSubscriber


class NoOpTracer:
    name = "noop"

    async def on_event(self, event: Event) -> None:
        return None
