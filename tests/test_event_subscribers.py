from __future__ import annotations

from typing import Any

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.message import ToolCall
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


class CapturingProvider:
    name = "capturing"

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = list(turns)
        self._index = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self._index >= len(self._turns):
            return LLMResponse(text="fallback", finish_reason="stop")
        turn = self._turns[self._index]
        self._index += 1
        return turn


class RecordingSubscriber:
    name = "recorder"

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def on_event(self, event: Event) -> None:
        self.events.append(event)


class ExplodingSubscriber:
    name = "exploding"

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def on_event(self, event: Event) -> None:
        self.events.append(event)
        raise RuntimeError("boom")


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
    return {"echo": args}


def _echo_tool(name: str = "echo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object"},
        handler=_echo_handler,
    )


def _session() -> Session:
    return Session(id="sess-sub", system_prompt="sys")


async def _collect(agen):
    return [event async for event in agen]


async def test_run_emits_run_start_and_run_end_bookends():
    provider = CapturingProvider([LLMResponse(text="ok", finish_reason="stop")])
    engine = Engine(provider, tools=[], config=EngineConfig(model="fake"))
    session = _session()

    events = await _collect(engine.run(session))

    assert events[0].type == EventType.RUN_START
    assert events[-1].type == EventType.RUN_END
    assert events[0].payload["model"] == "fake"
    assert events[-1].payload["final_state"] == "done"


async def test_subscriber_receives_every_event_in_order():
    provider = CapturingProvider(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={})],
                finish_reason="tool_use",
            ),
            LLMResponse(text="done", finish_reason="stop"),
        ]
    )
    subscriber = RecordingSubscriber()
    engine = Engine(
        provider,
        tools=[_echo_tool()],
        config=EngineConfig(model="fake"),
        event_subscribers=[subscriber],
    )
    session = _session()

    yielded = await _collect(engine.run(session))

    yielded_types = [e.type for e in yielded]
    captured_types = [e.type for e in subscriber.events]
    assert captured_types == yielded_types

    assert subscriber.events[0].type == EventType.RUN_START
    assert subscriber.events[-1].type == EventType.RUN_END
    assert EventType.STEP_START in captured_types
    assert EventType.LLM_CALL_START in captured_types
    assert EventType.LLM_CALL_END in captured_types
    assert EventType.TOOL_CALL_START in captured_types
    assert EventType.TOOL_CALL_END in captured_types
    assert EventType.STATE_CHANGED in captured_types


async def test_subscriber_exception_does_not_break_engine():
    provider = CapturingProvider([LLMResponse(text="ok", finish_reason="stop")])
    exploder = ExplodingSubscriber()
    recorder = RecordingSubscriber()
    engine = Engine(
        provider,
        tools=[],
        config=EngineConfig(model="fake"),
        event_subscribers=[exploder, recorder],
    )
    session = _session()

    events = await _collect(engine.run(session))

    assert events[-1].type == EventType.RUN_END
    assert any(e.type == EventType.RUN_START for e in recorder.events)
    assert any(e.type == EventType.RUN_END for e in recorder.events)
    assert len(exploder.events) == len(recorder.events)
