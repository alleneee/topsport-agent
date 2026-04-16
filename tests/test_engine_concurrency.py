from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from topsport_agent.engine.concurrency import EngineGuard, guarded_run
from topsport_agent.engine.interject_queue import InterjectQueue
from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.engine.loop_detector import LOOP_MESSAGE, LoopDetector
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


class ScriptedProvider:
    name = "scripted"

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = list(turns)
        self._index = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self._index >= len(self._turns):
            return LLMResponse(text="fallback")
        turn = self._turns[self._index]
        self._index += 1
        return turn


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
    return {"echo": args}


def _echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="echo",
        parameters={"type": "object"},
        handler=_echo_handler,
    )


def _session(sid: str = "s1") -> Session:
    return Session(id=sid, system_prompt="sys")


async def _collect(agen):
    return [event async for event in agen]


def test_loop_detector_detects_identical_calls():
    detector = LoopDetector(window=5, threshold=3)
    assert detector.check("s1", "echo", {"x": 1}) is False
    assert detector.check("s1", "echo", {"x": 1}) is False
    assert detector.check("s1", "echo", {"x": 1}) is True


def test_loop_detector_different_args_no_trigger():
    detector = LoopDetector(window=5, threshold=3)
    assert detector.check("s1", "echo", {"x": 1}) is False
    assert detector.check("s1", "echo", {"x": 2}) is False
    assert detector.check("s1", "echo", {"x": 3}) is False


def test_loop_detector_different_tools_no_trigger():
    detector = LoopDetector(window=5, threshold=3)
    detector.check("s1", "tool_a", {"x": 1})
    detector.check("s1", "tool_b", {"x": 1})
    assert detector.check("s1", "tool_a", {"x": 1}) is False


def test_loop_detector_session_scoped():
    detector = LoopDetector(threshold=2)
    detector.check("s1", "echo", {"x": 1})
    detector.check("s2", "echo", {"x": 1})
    assert detector.check("s1", "echo", {"x": 1}) is True
    assert detector.check("s2", "echo", {"x": 1}) is True


def test_loop_detector_clear_resets():
    detector = LoopDetector(threshold=2)
    detector.check("s1", "echo", {"x": 1})
    detector.clear("s1")
    assert detector.check("s1", "echo", {"x": 1}) is False


async def test_loop_detector_wrap_returns_loop_message():
    detector = LoopDetector(threshold=2)

    async def handler(args: dict[str, Any], ctx: ToolContext) -> Any:
        return {"ok": True}

    spec = ToolSpec(name="echo", description="", parameters={}, handler=handler)
    wrapped = detector.wrap(spec)

    cancel = asyncio.Event()
    ctx = ToolContext(session_id="s1", call_id="c1", cancel_event=cancel)

    result1 = await wrapped.handler({"x": 1}, ctx)
    assert result1 == {"ok": True}

    result2 = await wrapped.handler({"x": 1}, ctx)
    assert result2["loop_detected"] is True
    assert LOOP_MESSAGE in result2["message"]


async def test_loop_detector_wrap_passes_through_on_varied_args():
    detector = LoopDetector(threshold=3)
    spec = ToolSpec(name="echo", description="", parameters={}, handler=_echo_handler)
    wrapped = detector.wrap(spec)
    cancel = asyncio.Event()
    ctx = ToolContext(session_id="s1", call_id="c1", cancel_event=cancel)

    for i in range(5):
        result = await wrapped.handler({"x": i}, ctx)
        assert "loop_detected" not in result


async def test_interject_queue_enqueue_and_flush():
    q = InterjectQueue()
    await q.enqueue("s1", Message(role=Role.USER, content="interjected"))
    flushed = q.flush("s1")
    assert len(flushed) == 1
    assert flushed[0].content == "interjected"
    assert q.flush("s1") == []


async def test_interject_queue_session_scoped():
    q = InterjectQueue()
    await q.enqueue("s1", Message(role=Role.USER, content="for s1"))
    await q.enqueue("s2", Message(role=Role.USER, content="for s2"))
    assert len(q.flush("s1")) == 1
    assert len(q.flush("s2")) == 1
    assert q.flush("s1") == []


async def test_interject_queue_after_step_appends_to_session():
    q = InterjectQueue()
    session = _session()
    session.messages.append(Message(role=Role.USER, content="original"))
    await q.enqueue("s1", Message(role=Role.USER, content="injected"))

    await q.after_step(session, step=0)

    assert len(session.messages) == 2
    assert session.messages[1].content == "injected"


async def test_interject_queue_after_step_noop_when_empty():
    q = InterjectQueue()
    session = _session()
    session.messages.append(Message(role=Role.USER, content="original"))

    await q.after_step(session, step=0)

    assert len(session.messages) == 1


async def test_engine_guard_try_take_and_release():
    guard = EngineGuard()
    assert await guard.try_take("s1") is True
    assert guard.is_running("s1") is True
    assert await guard.try_take("s1") is False
    await guard.release("s1")
    assert guard.is_running("s1") is False
    assert await guard.try_take("s1") is True


async def test_engine_guard_concurrent_sessions_independent():
    guard = EngineGuard()
    assert await guard.try_take("s1") is True
    assert await guard.try_take("s2") is True
    assert await guard.try_take("s1") is False
    await guard.release("s1")
    assert await guard.try_take("s1") is True


async def test_guarded_run_releases_on_success():
    provider = ScriptedProvider([LLMResponse(text="ok")])
    engine = Engine(provider, tools=[], config=EngineConfig(model="fake"))
    guard = EngineGuard()
    session = _session()

    await _collect(guarded_run(engine, session, guard))

    assert session.state == RunState.DONE
    assert guard.is_running("s1") is False


async def test_guarded_run_releases_on_cancel():
    provider = ScriptedProvider([LLMResponse(text="slow")])

    async def slow_complete(request):
        await asyncio.sleep(5)
        return LLMResponse(text="slow")

    provider.complete = slow_complete

    engine = Engine(provider, tools=[], config=EngineConfig(model="fake"))
    guard = EngineGuard()
    session = _session()

    task = asyncio.create_task(_collect(guarded_run(engine, session, guard)))
    await asyncio.sleep(0.05)
    engine.cancel()
    await asyncio.wait_for(task, timeout=1.0)

    assert guard.is_running("s1") is False


async def test_guarded_run_rejects_concurrent_same_session():
    provider = ScriptedProvider([LLMResponse(text="slow")])

    async def slow_complete(request):
        await asyncio.sleep(5)
        return LLMResponse(text="slow")

    provider.complete = slow_complete

    engine = Engine(provider, tools=[], config=EngineConfig(model="fake"))
    guard = EngineGuard()

    session1 = _session("s1")
    session2 = _session("s1")

    task1 = asyncio.create_task(_collect(guarded_run(engine, session1, guard)))
    await asyncio.sleep(0.01)

    with pytest.raises(RuntimeError, match="already running"):
        await _collect(guarded_run(engine, session2, guard))

    engine.cancel()
    await asyncio.wait_for(task1, timeout=1.0)


async def test_interject_queue_as_post_step_hook_in_engine():
    q = InterjectQueue()
    provider = ScriptedProvider(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})],
                finish_reason="tool_use",
            ),
            LLMResponse(text="done"),
        ]
    )
    engine = Engine(
        provider,
        tools=[_echo_tool()],
        config=EngineConfig(model="fake"),
        post_step_hooks=[q],
    )
    session = _session()

    await q.enqueue("s1", Message(role=Role.USER, content="interjection"))

    await _collect(engine.run(session))

    assert any(m.content == "interjection" for m in session.messages)
    assert session.state == RunState.DONE
