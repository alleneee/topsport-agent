from __future__ import annotations

from ..engine.hooks import EventSubscriber
from ..types.events import Event

# Tracer 本质就是 EventSubscriber；类型别名让调用方语义更清晰。
Tracer = EventSubscriber


class NoOpTracer:
    """默认空实现：不装 langfuse 时引擎仍能正常运行，零开销。"""
    name = "noop"

    async def on_event(self, event: Event) -> None:
        return None
