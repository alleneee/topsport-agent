from __future__ import annotations

from typing import Protocol

from .types import MemoryEntry


class MemoryStore(Protocol):
    async def write(self, session_id: str, entry: MemoryEntry) -> None: ...

    async def read(self, session_id: str, key: str) -> MemoryEntry | None: ...

    async def list(self, session_id: str) -> list[MemoryEntry]: ...

    async def delete(self, session_id: str, key: str) -> bool: ...
