from __future__ import annotations

import asyncio
from collections import OrderedDict

from ..types.message import Message
from ..types.session import Session


class InterjectQueue:
    """工具执行期间外部可向队列投递消息，步骤结束后统一注入 session，避免并发写消息列表。"""

    name = "interject"

    def __init__(self, max_sessions: int = 1000) -> None:
        self._queues: OrderedDict[str, asyncio.Queue[Message]] = OrderedDict()
        self._max_sessions = max_sessions

    async def enqueue(self, session_id: str, message: Message) -> None:
        if session_id in self._queues:
            self._queues.move_to_end(session_id)
        queue = self._queues.setdefault(session_id, asyncio.Queue())
        if len(self._queues) > self._max_sessions:
            self._queues.popitem(last=False)
        await queue.put(message)

    def flush(self, session_id: str) -> list[Message]:
        """flush 是同步的：after_step 在引擎主循环中调用，不需要 await。"""
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
        """作为 PostStepHook 被引擎调用，把排队消息追加到本轮结尾。"""
        pending = self.flush(session.id)
        if pending:
            session.messages.extend(pending)
