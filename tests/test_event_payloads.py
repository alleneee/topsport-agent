"""验证 Event.typed_payload() 对每种 EventType 都工作，并用真实 engine 流量打底。

两层验证：
1. 单元测试：每个 EventType 手工构造 payload + typed_payload() 断言字段类型。
2. 集成测试：跑一遍 Engine + Orchestrator 的完整生命周期，收集所有 event，
   对每个 event 调一次 typed_payload()，确保 schema 和发布端同步、没字段漏注册。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.event_payloads import (
    EVENT_PAYLOAD_SCHEMAS,
    LLMCallEndPayload,
    PlanStepFailedPayload,
    RunStartPayload,
    StepEndPayload,
    TokenUsage,
    ToolCallEndPayload,
    ToolCallStartPayload,
)
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


# ---------------------------------------------------------------------------
# 单元：registry 完整性
# ---------------------------------------------------------------------------


def test_every_event_type_has_schema():
    """EventType 枚举和 schema 注册表必须一一对应，新加 EventType 忘记注册会在
    event_payloads 模块加载期就断言失败；这里再做一次 runtime 双保险。"""
    for et in EventType:
        assert et in EVENT_PAYLOAD_SCHEMAS, f"EventType {et.value} missing schema"


# ---------------------------------------------------------------------------
# 单元：每种 payload 的强类型访问
# ---------------------------------------------------------------------------


def test_run_start_payload_typed_access():
    ev = Event(
        type=EventType.RUN_START,
        session_id="s1",
        payload={
            "model": "claude-opus-4-7",
            "goal": "test",
            "initial_message_count": 3,
            "max_steps": 10,
        },
    )
    typed = ev.typed_payload()
    assert isinstance(typed, RunStartPayload)
    assert typed.model == "claude-opus-4-7"
    assert typed.max_steps == 10


def test_llm_call_end_payload_usage_is_dict():
    """usage 保留 dict，不强行拆成 TokenUsage，兼容 provider 各种扩展字段。"""
    ev = Event(
        type=EventType.LLM_CALL_END,
        session_id="s1",
        payload={
            "step": 0,
            "tool_call_count": 2,
            "finish_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read": 10},
        },
    )
    typed = ev.typed_payload()
    assert isinstance(typed, LLMCallEndPayload)
    assert typed.tool_call_count == 2
    assert typed.usage["cache_read"] == 10
    # 强类型版本：订阅者想要就单独 validate
    usage_typed = TokenUsage.model_validate(typed.usage)
    assert usage_typed.input_tokens == 100


def test_tool_call_payload_roundtrip():
    start_ev = Event(
        type=EventType.TOOL_CALL_START,
        session_id="s1",
        payload={"name": "bash", "call_id": "c1", "registered": True},
    )
    start_typed = start_ev.typed_payload()
    assert isinstance(start_typed, ToolCallStartPayload)
    assert start_typed.name == "bash"

    end_ev = Event(
        type=EventType.TOOL_CALL_END,
        session_id="s1",
        payload={"name": "bash", "call_id": "c1", "is_error": False},
    )
    end_typed = end_ev.typed_payload()
    assert isinstance(end_typed, ToolCallEndPayload)
    assert end_typed.is_error is False


def test_step_end_has_two_variants():
    """STEP_END 有两条发布路径：step vs reason；schema 宽松允许任一缺失。"""
    normal = Event(
        type=EventType.STEP_END, session_id="s1", payload={"step": 3}
    ).typed_payload()
    assert isinstance(normal, StepEndPayload)
    assert normal.step == 3 and normal.reason is None

    exhausted = Event(
        type=EventType.STEP_END,
        session_id="s1",
        payload={"reason": "max_steps_reached"},
    ).typed_payload()
    assert isinstance(exhausted, StepEndPayload)
    assert exhausted.step is None and exhausted.reason == "max_steps_reached"


def test_plan_step_failed_nested_entries():
    """PLAN_STEP_FAILED 的 failed_steps 是嵌套 model 列表，pydantic 自动递归校验。"""
    ev = Event(
        type=EventType.PLAN_STEP_FAILED,
        session_id="plan-1",
        payload={
            "plan_id": "plan-1",
            "failed_steps": [
                {"id": "s1", "error": "boom"},
                {"id": "s2", "error": None},
            ],
        },
    )
    typed = ev.typed_payload()
    assert isinstance(typed, PlanStepFailedPayload)
    assert len(typed.failed_steps) == 2
    assert typed.failed_steps[0].id == "s1"


def test_unknown_field_is_silently_ignored():
    """extra="ignore"：发布者多塞字段不报错，保证兼容发布侧的逐步演进。"""
    ev = Event(
        type=EventType.STATE_CHANGED,
        session_id="s1",
        payload={"state": "done", "extra_debug_field": "some-info"},
    )
    typed = ev.typed_payload()
    assert typed.state == "done"
    # 原 dict 仍然完整保留，旧路径继续工作
    assert ev.payload["extra_debug_field"] == "some-info"


def test_missing_required_field_raises():
    """发布者漏必选字段应该在 typed 访问时立刻暴露。"""
    ev = Event(
        type=EventType.ERROR,
        session_id="s1",
        payload={"kind": "ValueError"},  # 缺 message
    )
    with pytest.raises(ValidationError):
        ev.typed_payload()


def test_type_coercion_is_strict_on_fundamental_types():
    """bool 字段传字符串应报错——防止 payload 类型漂移。"""
    ev = Event(
        type=EventType.TOOL_CALL_END,
        session_id="s1",
        payload={"name": "bash", "call_id": "c1", "is_error": "not-a-bool"},
    )
    with pytest.raises(ValidationError):
        ev.typed_payload()


def test_payload_model_is_frozen():
    """payload schema 冻结：typed_payload() 返回值不可变，避免订阅者互相污染。"""
    ev = Event(
        type=EventType.STATE_CHANGED,
        session_id="s1",
        payload={"state": "running"},
    )
    typed = ev.typed_payload()
    with pytest.raises(ValidationError):
        typed.state = "done"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 集成：真实 Engine 流量每个 event 都能 typed_payload
# ---------------------------------------------------------------------------


class _ScriptedProvider:
    """按剧本逐步返回 LLMResponse：第一步调工具，第二步无工具收尾。

    走 duck-typing 而非继承 LLMProvider Protocol——Protocol 要求结构匹配即可，
    不继承避免 Pyright 把 Protocol 字段当成 abstract 逼着实现。
    """

    name = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> str:
    del ctx
    return f"echoed:{args.get('text', '')}"


@pytest.mark.asyncio
async def test_all_engine_events_validate():
    """跑一遍 Engine.run 完整流程，收集所有 Event，对每个调 typed_payload()。

    此测试保证：凡是 engine 会真实发出的 EventType，schema 都校验通过；
    防止发布侧悄悄加字段改字段时 schema 没同步。
    """
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="calling tool",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
                finish_reason="tool_use",
                usage={"input_tokens": 10, "output_tokens": 5},
                response_metadata=None,
            ),
            LLMResponse(
                text="done",
                tool_calls=[],
                finish_reason="end_turn",
                usage={"input_tokens": 15, "output_tokens": 3},
                response_metadata=None,
            ),
        ]
    )
    tool = ToolSpec(
        name="echo",
        description="echo back",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=_echo_handler,
    )
    engine = Engine(
        provider=provider,
        tools=[tool],
        config=EngineConfig(model="test-model", max_steps=5),
    )
    session = Session(id="s-it", system_prompt="test")
    session.messages.append(Message(role=Role.USER, content="go"))

    events: list[Event] = []
    async for ev in engine.run(session):
        events.append(ev)

    # 确认跑过的 event 类型覆盖关键生命周期
    seen_types = {ev.type for ev in events}
    for expected in (
        EventType.RUN_START,
        EventType.STEP_START,
        EventType.LLM_CALL_START,
        EventType.LLM_CALL_END,
        EventType.MESSAGE_APPENDED,
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_END,
        EventType.STATE_CHANGED,
        EventType.RUN_END,
    ):
        assert expected in seen_types, f"engine never emitted {expected.value}"

    # 所有 event 的 payload 都能强类型化
    for ev in events:
        typed = ev.typed_payload()
        assert typed is not None, f"event {ev.type.value} payload failed typing"
