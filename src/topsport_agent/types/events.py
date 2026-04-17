from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    """事件类型覆盖引擎全生命周期：run/step/llm/tool 四层嵌套 + plan 工作流。

    订阅者按 yield 顺序接收事件，单个订阅者异常不影响引擎和其他订阅者。
    """
    RUN_START = "run.start"
    RUN_END = "run.end"
    STEP_START = "step.start"
    STEP_END = "step.end"
    LLM_CALL_START = "llm.call.start"
    LLM_TEXT_DELTA = "llm.text.delta"
    LLM_CALL_END = "llm.call.end"
    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    MESSAGE_APPENDED = "message.appended"
    STATE_CHANGED = "state.changed"
    ERROR = "error"
    CANCELLED = "cancelled"
    PLAN_CREATED = "plan.created"
    PLAN_APPROVED = "plan.approved"
    PLAN_REJECTED = "plan.rejected"
    PLAN_STEP_START = "plan.step.start"
    PLAN_STEP_END = "plan.step.end"
    PLAN_STEP_FAILED = "plan.step.failed"
    PLAN_WAITING = "plan.waiting"
    PLAN_DONE = "plan.done"
    PLAN_FAILED = "plan.failed"


@dataclass(slots=True)
class Event:
    """Event 是不可变的事件信封：类型 + 会话归属 + 自由 payload。

    payload 内容约定由各 EventType 隐式定义，订阅者按类型解读。
    """
    type: EventType
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
