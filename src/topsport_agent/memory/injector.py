from __future__ import annotations

from ..types.message import Message, Role
from ..types.session import Session
from .store import MemoryStore
from .types import MemoryType


class MemoryInjector:
    """ContextProvider 实现：每步 LLM 调用前重新读取记忆并注入为临时 system 消息。

    输出是短暂的——只参与本次调用，不写入 session.messages，避免记忆膨胀。
    section_tag/section_priority 供 PromptBuilder 按标签分区组装。
    """
    name = "memory"
    section_tag = "working-memory"
    section_priority = 200

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
        # 可选的类型过滤：调用方可只注入 goal/constraint 等子集，控制上下文长度
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
        return [Message(
            role=Role.SYSTEM,
            content=body,
            extra={
                "section_tag": self.section_tag,
                "section_priority": self.section_priority,
            },
        )]
