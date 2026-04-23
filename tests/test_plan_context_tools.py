"""plan_context_read / plan_context_merge 工具 + Orchestrator 注入。"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated

import pytest

from topsport_agent.engine.orchestrator import Orchestrator, SubAgentConfig
from topsport_agent.engine.plan_context_tools import (
    PlanContextBridge,
    PlanContextToolSource,
)
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.message import ToolCall
from topsport_agent.types.plan import Plan, PlanStep, StepStatus
from topsport_agent.types.plan_context import PlanContext, Reducer
from topsport_agent.types.tool import ToolContext


class MyCtx(PlanContext):
    notes: Annotated[list[str], Reducer(operator.add)] = []
    quality: float = 0.0


def _make_plan(ctx: MyCtx | None) -> Plan:
    return Plan(
        id="p",
        goal="g",
        steps=[PlanStep(id="a", title="A", instructions="")],
        context=ctx,
    )


def _tool_ctx() -> ToolContext:
    return ToolContext(session_id="s", call_id="c", cancel_event=asyncio.Event())


# ---------------------------------------------------------------------------
# Bridge: read / merge
# ---------------------------------------------------------------------------


async def test_read_returns_empty_dict_when_no_context() -> None:
    plan = _make_plan(None)
    bridge = PlanContextBridge(plan)
    assert await bridge.read() == {}


async def test_read_returns_current_snapshot() -> None:
    plan = _make_plan(MyCtx(notes=["hi"], quality=0.5))
    bridge = PlanContextBridge(plan)
    snap = await bridge.read()
    assert snap == {"notes": ["hi"], "quality": 0.5}


async def test_merge_applies_reducer_and_updates_plan_context() -> None:
    plan = _make_plan(MyCtx(notes=["a"]))
    bridge = PlanContextBridge(plan)
    result = await bridge.merge("notes", ["b"])
    assert result["notes"] == ["a", "b"]
    assert plan.context is not None
    assert plan.context.notes == ["a", "b"]   # type: ignore[attr-defined]


async def test_merge_without_reducer_overrides() -> None:
    plan = _make_plan(MyCtx(quality=0.0))
    bridge = PlanContextBridge(plan)
    await bridge.merge("quality", 0.9)
    assert plan.context is not None
    assert plan.context.quality == 0.9    # type: ignore[attr-defined]


async def test_merge_without_context_raises() -> None:
    plan = _make_plan(None)
    bridge = PlanContextBridge(plan)
    with pytest.raises(ValueError) as ei:
        await bridge.merge("notes", ["x"])
    assert "no PlanContext" in str(ei.value)


async def test_merge_unknown_field_raises_key_error() -> None:
    plan = _make_plan(MyCtx())
    bridge = PlanContextBridge(plan)
    with pytest.raises(KeyError):
        await bridge.merge("nonexistent", 1)


async def test_concurrent_merges_do_not_lose_updates() -> None:
    """并发 merge 同一字段不丢更新：lock 串行化 read-modify-write。"""
    plan = _make_plan(MyCtx(notes=[]))
    bridge = PlanContextBridge(plan)

    async def _append(val: str) -> None:
        await bridge.merge("notes", [val])

    await asyncio.gather(*[_append(f"n{i}") for i in range(50)])
    assert plan.context is not None
    assert len(plan.context.notes) == 50                                         # type: ignore[attr-defined]
    assert set(plan.context.notes) == {f"n{i}" for i in range(50)}               # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ToolSpec handlers (模拟 engine 调用)
# ---------------------------------------------------------------------------


async def test_read_tool_returns_snapshot() -> None:
    plan = _make_plan(MyCtx(notes=["x"]))
    bridge = PlanContextBridge(plan)
    tools = bridge.make_tools()
    read_tool = next(t for t in tools if t.name == "plan_context_read")
    result = await read_tool.handler({}, _tool_ctx())
    assert result == {"notes": ["x"], "quality": 0.0}


async def test_merge_tool_applies_reducer() -> None:
    plan = _make_plan(MyCtx(notes=[]))
    bridge = PlanContextBridge(plan)
    tools = bridge.make_tools()
    merge_tool = next(t for t in tools if t.name == "plan_context_merge")

    result = await merge_tool.handler(
        {"key": "notes", "value": ["hello"]}, _tool_ctx()
    )
    assert result == {
        "merged": True,
        "field": "notes",
        "context": {"notes": ["hello"], "quality": 0.0},
    }


async def test_merge_tool_missing_argument_raises() -> None:
    plan = _make_plan(MyCtx())
    bridge = PlanContextBridge(plan)
    merge_tool = bridge.make_merge_tool()
    with pytest.raises(ValueError):
        await merge_tool.handler({"key": "notes"}, _tool_ctx())   # missing value


# ---------------------------------------------------------------------------
# ToolSource integration
# ---------------------------------------------------------------------------


async def test_tool_source_lists_both_tools() -> None:
    bridge = PlanContextBridge(_make_plan(MyCtx()))
    source = PlanContextToolSource(bridge)
    tools = await source.list_tools()
    names = sorted(t.name for t in tools)
    assert names == ["plan_context_merge", "plan_context_read"]


async def test_tool_source_name_is_stable() -> None:
    source = PlanContextToolSource(PlanContextBridge(_make_plan(MyCtx())))
    assert source.name == "plan_context"


# ---------------------------------------------------------------------------
# Orchestrator 集成：sub-agent 真实调用这些工具
# ---------------------------------------------------------------------------


class MergingProvider:
    """模拟 sub-agent：第 1 次 LLM 调用返回 plan_context_merge 工具调用；
    第 2 次返回纯文本。两步完成一个 ReAct 循环。
    """

    name = "merging"

    def __init__(self, key: str, value: object) -> None:
        self._key = key
        self._value = value
        self.call_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                text="",
                finish_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="plan_context_merge",
                        arguments={"key": self._key, "value": self._value},
                    )
                ],
            )
        return LLMResponse(text="done", finish_reason="stop")


class _Collector:
    """Event subscriber 捕获子引擎 TOOL_CALL_END 等内部事件——orchestrator 的
    execute() 只 yield plan 层事件，子引擎事件要靠 subscriber 透传。"""

    name = "collector"

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def on_event(self, event: Event) -> None:
        self.events.append(event)


async def test_orchestrator_injects_tools_when_context_present() -> None:
    """plan 带 context → sub-agent 能成功调用 plan_context_merge 把结果写进共享 ctx。"""
    plan = _make_plan(MyCtx(notes=["seed"]))
    provider = MergingProvider("notes", ["delta"])
    collector = _Collector()
    orch = Orchestrator(
        plan,
        SubAgentConfig(provider=provider, model="fake"),
        event_subscribers=[collector],
    )
    async for _ in orch.execute():
        pass

    # 有 tool_call_end 事件且成功
    tool_ends = [
        e for e in collector.events if e.type == EventType.TOOL_CALL_END
    ]
    merge_ends = [
        e for e in tool_ends if e.payload.get("name") == "plan_context_merge"
    ]
    assert len(merge_ends) == 1
    assert merge_ends[0].payload["is_error"] is False

    # context 被合并
    assert plan.context is not None
    assert plan.context.notes == ["seed", "delta"]    # type: ignore[attr-defined]
    # step 正常完成
    assert plan.step_by_id("a").status == StepStatus.DONE   # type: ignore[union-attr]


async def test_orchestrator_skips_tool_injection_when_no_context() -> None:
    """无 context → sub-agent 工具池不含 plan_context_*；调用会报 tool 未注册。"""
    plan = _make_plan(None)
    provider = MergingProvider("notes", ["x"])
    collector = _Collector()
    orch = Orchestrator(
        plan,
        SubAgentConfig(provider=provider, model="fake"),
        event_subscribers=[collector],
    )
    async for _ in orch.execute():
        pass

    # tool 未注册 → engine 返回 is_error=True
    tool_ends = [
        e for e in collector.events if e.type == EventType.TOOL_CALL_END
    ]
    merge_ends = [
        e for e in tool_ends if e.payload.get("name") == "plan_context_merge"
    ]
    assert len(merge_ends) == 1
    assert merge_ends[0].payload["is_error"] is True
