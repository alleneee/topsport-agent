"""会话存储：按 session_id 缓存 (Agent, Session)，线程/任务安全。

服务端有状态：同一 session_id 的多轮对话共享消息历史。
Agent 在首次请求时按 model 构造，后续请求复用；session_id 未指定时自动生成。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from ..agent.base import Agent
from ..llm.provider import LLMProvider
from ..types.session import Session


@dataclass(slots=True)
class SessionEntry:
    agent: Agent
    session: Session
    lock: asyncio.Lock
    created_at: float
    last_used_at: float


AgentFactory = Callable[[LLMProvider, str], Agent]


class SessionStore:
    """简单 LRU + TTL 的内存会话存储。

    - `get_or_create` 返回 (entry, is_new)。
    - 每个 session 自带一把 lock，同一会话内请求串行化，避免 messages 乱序。
    - 容量达到 max_sessions 时先 evict 最久未使用；evict 时调 agent.close。
    - ttl 过期的条目下次 get_or_create 前也会被清理。
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        provider: LLMProvider,
        max_sessions: int = 128,
        ttl_seconds: int = 3600,
    ) -> None:
        self._agent_factory = agent_factory
        self._provider = provider
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._entries: dict[str, SessionEntry] = {}
        self._global_lock = asyncio.Lock()

    async def get_or_create(
        self, session_id: str | None, model: str
    ) -> tuple[str, SessionEntry, bool]:
        async with self._global_lock:
            await self._evict_expired_locked()
            sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
            entry = self._entries.get(sid)
            if entry is not None:
                entry.last_used_at = time.monotonic()
                return sid, entry, False

            await self._evict_if_full_locked()
            agent = self._agent_factory(self._provider, model)
            session = agent.new_session(sid)
            now = time.monotonic()
            entry = SessionEntry(
                agent=agent,
                session=session,
                lock=asyncio.Lock(),
                created_at=now,
                last_used_at=now,
            )
            self._entries[sid] = entry
            return sid, entry, True

    async def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, e in self._entries.items() if now - e.last_used_at > self._ttl]
        for sid in expired:
            await self._close_entry(sid)

    async def _evict_if_full_locked(self) -> None:
        if len(self._entries) < self._max:
            return
        oldest_sid = min(self._entries, key=lambda s: self._entries[s].last_used_at)
        await self._close_entry(oldest_sid)

    async def _close_entry(self, sid: str) -> None:
        entry = self._entries.pop(sid, None)
        if entry is None:
            return
        try:
            await entry.agent.close()
        except Exception:
            pass

    async def close_all(self) -> None:
        async with self._global_lock:
            for sid in list(self._entries):
                await self._close_entry(sid)

    async def get(self, sid: str) -> SessionEntry | None:
        async with self._global_lock:
            entry = self._entries.get(sid)
            if entry is not None:
                entry.last_used_at = time.monotonic()
            return entry

    async def delete(self, sid: str) -> bool:
        """按 sid 删除 session。找到返回 True，未找到 False。"""
        async with self._global_lock:
            if sid not in self._entries:
                return False
            await self._close_entry(sid)
            return True

    async def ids_with_prefix(self, prefix: str) -> list[str]:
        """列出以给定前缀开头的 session id（用于按 principal 过滤）。"""
        async with self._global_lock:
            return sorted(sid for sid in self._entries if sid.startswith(prefix))
