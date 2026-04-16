from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    RUN_START = "run.start"
    RUN_END = "run.end"
    STEP_START = "step.start"
    STEP_END = "step.end"
    LLM_CALL_START = "llm.call.start"
    LLM_CALL_END = "llm.call.end"
    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    MESSAGE_APPENDED = "message.appended"
    STATE_CHANGED = "state.changed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Event:
    type: EventType
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
