"""会话存储：按 session_id 缓存 (Agent, Session)，线程/任务安全。

服务端有状态：同一 session_id 的多轮对话共享消息历史。
Agent 在首次请求时按 model 构造，后续请求复用；session_id 未指定时自动生成。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..agent.base import Agent
from ..llm.provider import LLMProvider
from ..types.session import Session

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionEntry:
    agent: Agent
    session: Session
    lock: asyncio.Lock
    created_at: float
    last_used_at: float


AgentFactory = Callable[[LLMProvider, str], Agent]

# 钩子签名：(session_id, entry) -> awaitable[None]
# 触发时机：
#   on_session_created：session 首次创建时（get_or_create 里 is_new=True 分支）
#   on_session_closed：session 被 evict / ttl 超时 / delete / close_all 时，
#                       在 agent.close() 之前调用
# 典型用途：绑定 sandbox / 审计 session 生命周期。
SessionLifecycleHook = Callable[[str, "SessionEntry"], Awaitable[None]]
# 兼容旧导出名
SessionCloseHook = SessionLifecycleHook


class SessionStore:
    """简单 LRU + TTL 的内存会话存储。

    - `get_or_create` 返回 (entry, is_new)；可选传 tenant_id / principal 注入到 Session。
    - 每个 session 自带一把 lock，同一会话内请求串行化，避免 messages 乱序。
    - 容量达到 max_sessions 时先 evict 最久未使用；evict 时调 agent.close。
    - ttl 过期的条目下次 get_or_create 前也会被清理。
    - on_session_closed 钩子在 entry 被关闭前触发（一个钩子失败不影响其他）。
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        provider: LLMProvider,
        max_sessions: int = 128,
        ttl_seconds: int = 3600,
        on_session_created: list[SessionLifecycleHook] | None = None,
        on_session_closed: list[SessionLifecycleHook] | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._provider = provider
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._entries: dict[str, SessionEntry] = {}
        self._global_lock = asyncio.Lock()
        self._create_hooks: list[SessionLifecycleHook] = list(on_session_created or [])
        self._close_hooks: list[SessionLifecycleHook] = list(on_session_closed or [])

    def add_create_hook(self, hook: SessionLifecycleHook) -> None:
        """运行时追加 create 钩子。对已创建的 session 不追溯。"""
        self._create_hooks.append(hook)

    def add_close_hook(self, hook: SessionLifecycleHook) -> None:
        """运行时追加 close 钩子。对已关闭的 session 不追溯。"""
        self._close_hooks.append(hook)

    async def get_or_create(
        self,
        session_id: str | None,
        model: str,
        *,
        tenant_id: str | None = None,
        principal: str | None = None,
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
            # Async path：agent.new_session_async 解析 persona → granted_permissions，
            # 是 capability-ACL 的唯一正确入口。此前版本走 sync new_session，persona
            # 配置永远不生效（即 codex 指出的 P0-1 "断路" 问题）。
            # 兼容鸭子类型的测试 FakeAgent（仅实现 new_session）：缺 async 方法
            # 时退化到 sync 路径，保证 server 之外的集成点不破。
            new_async = getattr(agent, "new_session_async", None)
            if new_async is not None:
                session = await new_async(sid)
            else:
                session = agent.new_session(sid)
            # Server-provided tenant/principal always overrides whatever
            # AgentConfig.tenant_id seeded — server 明确传入的优先级更高。
            session.tenant_id = tenant_id
            session.principal = principal
            now = time.monotonic()
            entry = SessionEntry(
                agent=agent,
                session=session,
                lock=asyncio.Lock(),
                created_at=now,
                last_used_at=now,
            )
            self._entries[sid] = entry
            # create 钩子在 entry 已入 dict 后触发；单钩子失败不影响其他。
            # 持 _global_lock 调 hook 会阻塞其他请求，但钩子通常轻量（sandbox bind_tenant 只写 dict）
            for hook in self._create_hooks:
                try:
                    await hook(sid, entry)
                except Exception as exc:
                    _logger.warning(
                        "session create hook %r failed for sid=%s: %r",
                        getattr(hook, "__qualname__", hook), sid, exc,
                        extra={
                            "event": "session_create_hook_failed",
                            "session_id": sid,
                            "tenant_id": tenant_id,
                            "principal": principal,
                            "hook": getattr(hook, "__qualname__", str(hook)),
                        },
                    )
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
        # 先跑 close 钩子（如释放 sandbox），再 close agent；
        # 钩子异常互相隔离，单个钩子失败不影响其他钩子与 agent.close。
        for hook in self._close_hooks:
            try:
                await hook(sid, entry)
            except Exception as exc:
                _logger.warning(
                    "session close hook %r failed for sid=%s: %r",
                    getattr(hook, "__qualname__", hook), sid, exc,
                    extra={
                        "event": "session_close_hook_failed",
                        "session_id": sid,
                        "hook": getattr(hook, "__qualname__", str(hook)),
                    },
                )
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
