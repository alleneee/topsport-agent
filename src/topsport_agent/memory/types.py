from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MemoryType(StrEnum):
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
