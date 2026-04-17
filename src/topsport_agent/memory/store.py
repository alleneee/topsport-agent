from __future__ import annotations

from typing import Protocol

from .types import MemoryEntry


class MemoryStore(Protocol):
    """存储层以 Protocol 定义，实现方只需满足鸭子类型即可接入，无需继承。

    所有操作以 session_id 隔离，保证会话间记忆互不可见。
    """
    async def write(self, session_id: str, entry: MemoryEntry) -> None: ...

    async def read(self, session_id: str, key: str) -> MemoryEntry | None: ...

    async def list(self, session_id: str) -> list[MemoryEntry]: ...

    async def delete(self, session_id: str, key: str) -> bool: ...
