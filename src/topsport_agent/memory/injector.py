from __future__ import annotations

from ..types.message import Message, Role
from ..types.session import Session
from .store import MemoryStore
from .types import MemoryType


class MemoryInjector:
    name = "memory"

    def __init__(
        self,
        store: MemoryStore,
        types: list[MemoryType] | None = None,
        header: str = "Working memory",
    ) -> None:
        self._store = store
        self._types = types
        self._header = header

    async def provide(self, session: Session) -> list[Message]:
        entries = await self._store.list(session.id)
        if self._types:
            wanted = set(self._types)
            entries = [entry for entry in entries if entry.type in wanted]
        if not entries:
            return []
        blocks: list[str] = []
        for entry in entries:
            header_line = f"[{entry.type.value}] {entry.name}"
            if entry.description:
                header_line += f" — {entry.description}"
            blocks.append(f"{header_line}\n{entry.content}")
        body = f"## {self._header}\n\n" + "\n\n".join(blocks)
        return [Message(role=Role.SYSTEM, content=body)]
