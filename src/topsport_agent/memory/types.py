from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MemoryType(StrEnum):
    """记忆分类：语义标签决定注入优先级和 compaction 策略，而非纯粹的分组标签。"""
    GOAL = "goal"
    IDENTITY = "identity"
    FACT = "fact"
    CONSTRAINT = "constraint"
    NOTE = "note"


@dataclass(slots=True)
class MemoryEntry:
    key: str
    name: str
    description: str
    type: MemoryType
    content: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
