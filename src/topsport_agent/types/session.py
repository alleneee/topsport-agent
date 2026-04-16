from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .message import Message


class RunState(StrEnum):
    IDLE = "idle"
    WAITING_USER = "waiting_user"
    WAITING_CONFIRM = "waiting_confirm"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass(slots=True)
class Session:
    id: str
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    state: RunState = RunState.IDLE
    goal: str | None = None
