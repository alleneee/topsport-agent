"""Per-EventType pydantic payload schemas.

设计选择：
- 不替换 Event.payload 现状（dict[str, Any]）——20+ 订阅者 / CLI / server 都在读，一刀切风险大。
- 并行提供强类型层：每种 EventType 绑一个 BaseModel，通过 registry 索引。
- Event.typed_payload() 在需要类型安全的订阅者（metrics/tracer/server）里调用；旧代码不动。
- extra="ignore"：老的 payload 多塞了字段不会报错；新 schema 只看声明字段；
  即"schema 是订阅者看得到的最小合同"，发布者想塞更多 debug 字段仍可。
- 所有字段尽量 Optional 兜底；引擎 emit 时偶有字段缺（如 STEP_END 的 reason 路径），
  schema 宁可宽松放行也不要 runtime 错，毕竟兼容优先。

参考 claude-code/src/types/hooks.ts 的 Zod discriminatedUnion 设计：
每个子变体用 literal type 做判别，字段按 variant 按需定义。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .events import EventType

__all__ = [
    "EVENT_PAYLOAD_SCHEMAS",
    "EventPayload",
    "RunStartPayload",
    "RunEndPayload",
    "StepStartPayload",
    "StepEndPayload",
    "LLMCallStartPayload",
    "LLMTextDeltaPayload",
    "LLMCallEndPayload",
    "ToolCallStartPayload",
    "ToolCallEndPayload",
    "MessageAppendedPayload",
    "StateChangedPayload",
    "ErrorPayload",
    "CancelledPayload",
    "PlanApprovedPayload",
    "PlanCreatedPayload",
    "PlanRejectedPayload",
    "PlanStepStartPayload",
    "PlanStepEndPayload",
    "PlanStepFailedPayload",
    "PlanStepSkippedPayload",
    "PlanStepLoopPayload",
    "PlanWaitingPayload",
    "PlanDonePayload",
    "PlanFailedPayload",
    "TokenUsage",
]


class _BaseEventPayload(BaseModel):
    """所有 payload schema 的共同配置：忽略多余字段，兼容发布者塞 debug 数据。"""

    model_config = ConfigDict(extra="ignore", frozen=True)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class RunStartPayload(_BaseEventPayload):
    model: str
    goal: str | None = None
    initial_message_count: int = 0
    max_steps: int = 0


class RunEndPayload(_BaseEventPayload):
    final_state: str
    message_count: int = 0


# ---------------------------------------------------------------------------
# Step lifecycle
# ---------------------------------------------------------------------------


class StepStartPayload(_BaseEventPayload):
    step: int


class StepEndPayload(_BaseEventPayload):
    """STEP_END 有两条发布路径：带 step 的正常结束，带 reason 的 max_steps 结束。
    两个字段都可选，消费方按存在性判断。"""

    step: int | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


class LLMCallStartPayload(_BaseEventPayload):
    step: int
    model: str
    tool_count: int = 0
    ephemeral_msg_count: int = 0
    call_msg_count: int = 0
    stream: bool = False


class LLMTextDeltaPayload(_BaseEventPayload):
    step: int
    delta: str


class TokenUsage(_BaseEventPayload):
    """跨 provider 兼容的 usage 子 schema。Anthropic/OpenAI 字段都允许缺。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class LLMCallEndPayload(_BaseEventPayload):
    step: int
    tool_call_count: int = 0
    finish_reason: str | None = None
    # usage 保留 dict 兼容 provider 任意扩展字段；强类型版本见 TokenUsage.model_validate(usage)
    usage: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool call
# ---------------------------------------------------------------------------


class ToolCallStartPayload(_BaseEventPayload):
    name: str
    call_id: str
    registered: bool = True


class ToolCallEndPayload(_BaseEventPayload):
    name: str
    call_id: str
    is_error: bool = False


# ---------------------------------------------------------------------------
# Message / state / error
# ---------------------------------------------------------------------------


class MessageAppendedPayload(_BaseEventPayload):
    """两条发布路径：assistant 消息带 tool_call_count，tool 消息带 call_id。"""

    role: str
    tool_call_count: int | None = None
    call_id: str | None = None


class StateChangedPayload(_BaseEventPayload):
    state: str


class ErrorPayload(_BaseEventPayload):
    kind: str
    message: str


class CancelledPayload(_BaseEventPayload):
    """Engine 侧 CANCELLED 发空 dict；Orchestrator 侧带 plan_id。"""

    plan_id: str | None = None


# ---------------------------------------------------------------------------
# Plan lifecycle
# ---------------------------------------------------------------------------


class PlanCreatedPayload(_BaseEventPayload):
    """Planner 侧 emit，payload 结构当前由外部决定；保留 plan_id 兜底字段。"""

    plan_id: str | None = None
    goal: str | None = None
    step_count: int | None = None


class PlanApprovedPayload(_BaseEventPayload):
    plan_id: str
    goal: str
    step_count: int


class PlanRejectedPayload(_BaseEventPayload):
    plan_id: str | None = None
    reason: str | None = None


class PlanStepStartPayload(_BaseEventPayload):
    plan_id: str
    step_id: str
    title: str
    iteration: int = 1


class PlanStepEndPayload(_BaseEventPayload):
    plan_id: str
    step_id: str
    status: str
    result: str | None = None
    error: str | None = None
    iterations: int = 0


class _FailedStepEntry(_BaseEventPayload):
    id: str
    error: str | None = None


class PlanStepFailedPayload(_BaseEventPayload):
    plan_id: str
    failed_steps: list[_FailedStepEntry] = Field(default_factory=list)


class PlanStepSkippedPayload(_BaseEventPayload):
    plan_id: str
    step_id: str
    reason: Literal["condition_false", "condition_error"] | str
    error: str | None = None


class PlanStepLoopPayload(_BaseEventPayload):
    plan_id: str
    step_id: str
    iteration: int
    max_iterations: int
    reset_dependents: list[str] = Field(default_factory=list)


class PlanWaitingPayload(_BaseEventPayload):
    plan_id: str
    options: list[str] = Field(default_factory=list)


class PlanDonePayload(_BaseEventPayload):
    plan_id: str
    results: dict[str, str | None] = Field(default_factory=dict)


class PlanFailedPayload(_BaseEventPayload):
    plan_id: str
    reason: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EventPayload = _BaseEventPayload

EVENT_PAYLOAD_SCHEMAS: dict[EventType, type[_BaseEventPayload]] = {
    EventType.RUN_START: RunStartPayload,
    EventType.RUN_END: RunEndPayload,
    EventType.STEP_START: StepStartPayload,
    EventType.STEP_END: StepEndPayload,
    EventType.LLM_CALL_START: LLMCallStartPayload,
    EventType.LLM_TEXT_DELTA: LLMTextDeltaPayload,
    EventType.LLM_CALL_END: LLMCallEndPayload,
    EventType.TOOL_CALL_START: ToolCallStartPayload,
    EventType.TOOL_CALL_END: ToolCallEndPayload,
    EventType.MESSAGE_APPENDED: MessageAppendedPayload,
    EventType.STATE_CHANGED: StateChangedPayload,
    EventType.ERROR: ErrorPayload,
    EventType.CANCELLED: CancelledPayload,
    EventType.PLAN_CREATED: PlanCreatedPayload,
    EventType.PLAN_APPROVED: PlanApprovedPayload,
    EventType.PLAN_REJECTED: PlanRejectedPayload,
    EventType.PLAN_STEP_START: PlanStepStartPayload,
    EventType.PLAN_STEP_END: PlanStepEndPayload,
    EventType.PLAN_STEP_FAILED: PlanStepFailedPayload,
    EventType.PLAN_STEP_SKIPPED: PlanStepSkippedPayload,
    EventType.PLAN_STEP_LOOP: PlanStepLoopPayload,
    EventType.PLAN_WAITING: PlanWaitingPayload,
    EventType.PLAN_DONE: PlanDonePayload,
    EventType.PLAN_FAILED: PlanFailedPayload,
}

# 编译期断言：每个 EventType 都有对应 schema，防止新增 EventType 时忘记注册。
_MISSING = set(EventType) - set(EVENT_PAYLOAD_SCHEMAS)
assert not _MISSING, f"EventType missing payload schema: {sorted(e.value for e in _MISSING)}"
