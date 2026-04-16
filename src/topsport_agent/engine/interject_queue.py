from __future__ import annotations

import asyncio

from ..types.message import Message
from ..types.session import Session


class InterjectQueue:
    name = "interject"

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Message]] = {}

    async def enqueue(self, session_id: str, message: Message) -> None:
        queue = self._queues.setdefault(session_id, asyncio.Queue())
        await queue.put(message)

    def flush(self, session_id: str) -> list[Message]:
        queue = self._queues.get(session_id)
        if not queue:
            return []
        flushed: list[Message] = []
        while not queue.empty():
            flushed.append(queue.get_nowait())
        return flushed

    def clear(self, session_id: str) -> None:
        self._queues.pop(session_id, None)

    async def after_step(self, session: Session, step: int) -> None:
        pending = self.flush(session.id)
        if pending:
            session.messages.extend(pending)
