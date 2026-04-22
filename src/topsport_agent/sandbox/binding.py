"""把 OpenSandboxPool 的生命周期挂到 SessionStore。

使用方式（server 启动时）：

    pool = OpenSandboxPool.from_config(...)
    binding = SessionSandboxBinding(pool)
    store = SessionStore(
        agent_factory=..., provider=...,
        on_session_closed=[binding.on_session_closed],
    )

之后 session 被 evict / delete / ttl 超时时，绑定的沙箱会自动 release，
避免泄漏沙箱容器到 OpenSandbox server 里。
"""
from __future__ import annotations

import logging
from typing import Any

from .pool import OpenSandboxPool

_logger = logging.getLogger(__name__)


class SessionSandboxBinding:
    """SessionStore 生命周期钩子：

    - on_session_created：把 session 绑定到 tenant（供 pool 按 tenant 限额）
    - on_session_closed：释放 pool 里的沙箱 + 归还配额

    对从未 acquire 过 sandbox 的 session 无副作用（pool.release 幂等）。
    """

    def __init__(self, pool: OpenSandboxPool) -> None:
        self._pool = pool

    async def on_session_created(self, session_id: str, entry: Any) -> None:
        """session 新建时把 entry.session.tenant_id 写入 pool 的 tenant 映射。

        这样 tool_source 调 pool.acquire(sid) 时 pool 能按 tenant 取信号量，
        而 tool_source 本身不需要感知 tenant 概念（保持 ToolContext 最小）。
        """
        tenant_id = getattr(getattr(entry, "session", None), "tenant_id", None)
        try:
            self._pool.bind_tenant(session_id, tenant_id)
        except Exception as exc:
            _logger.warning(
                "bind_tenant failed sid=%s tenant=%s: %r",
                session_id, tenant_id, exc,
            )

    async def on_session_closed(self, session_id: str, entry: Any) -> None:
        """session 关闭时释放 sandbox（release 幂等）。"""
        del entry  # 未来可读 entry.session.tenant_id 做审计
        try:
            await self._pool.release(session_id)
        except Exception as exc:
            _logger.warning(
                "sandbox release failed on session close sid=%s: %r",
                session_id, exc,
            )
