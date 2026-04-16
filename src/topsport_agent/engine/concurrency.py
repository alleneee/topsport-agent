from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..types.events import Event
from ..types.session import Session


class EngineGuard:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._busy: dict[str, bool] = {}

    async def try_take(self, session_id: str) -> bool:
        async with self._lock:
            if self._busy.get(session_id, False):
                return False
            self._busy[session_id] = True
            return True

    async def release(self, session_id: str) -> None:
        async with self._lock:
            self._busy[session_id] = False

    def is_running(self, session_id: str) -> bool:
        return self._busy.get(session_id, False)


async def guarded_run(
    engine: object,
    session: Session,
    guard: EngineGuard,
) -> AsyncIterator[Event]:
    taken = await guard.try_take(session.id)
    if not taken:
        raise RuntimeError(f"session '{session.id}' is already running")
    try:
        async for event in engine.run(session):  # type: ignore[union-attr]
            yield event
    finally:
        await guard.release(session.id)
