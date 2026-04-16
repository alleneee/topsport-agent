from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import (
    LLM_RESPONSE_EXTRA_KEY,
    ProviderResponseMetadata,
    get_response_metadata,
)
from topsport_agent.types.events import EventType
from topsport_agent.types.message import Role, ToolCall
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


@dataclass
class ScriptedTurn:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    response_metadata: ProviderResponseMetadata | None = None
    delay: float = 0.0


class ScriptedProvider:
    name = "scripted"

    def __init__(self, turns: list[ScriptedTurn]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.seen_requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.seen_requests.append(request)
        if self._index >= len(self._turns):
            return LLMResponse(text="fallback", finish_reason="stop")
        turn = self._turns[self._index]
        self._index += 1
        if turn.delay:
            await asyncio.sleep(turn.delay)
        return LLMResponse(
            text=turn.text,
            tool_calls=list(turn.tool_calls),
            finish_reason="tool_use" if turn.tool_calls else "stop",
            response_metadata=turn.response_metadata,
        )


def _session() -> Session:
    return Session(id="s1", system_prompt="you are a test agent")


async def _collect(agen):
    return [event async for event in agen]


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
    return {"echoed": args}


def _echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="echo back arguments",
        parameters={"type": "object"},
        handler=_echo_handler,
    )


async def test_completes_when_no_tool_calls():
    provider = ScriptedProvider([ScriptedTurn(text="hello")])
    engine = Engine(provider, tools=[], config=EngineConfig(model="fake"))
    session = _session()

    events = await _collect(engine.run(session))

    assert session.state == RunState.DONE
    assert session.messages[-1].role == Role.ASSISTANT
    assert session.messages[-1].content == "hello"
    assert any(e.type == EventType.LLM_CALL_END for e in events)
    state_events = [e for e in events if e.type == EventType.STATE_CHANGED]
    assert state_events[0].payload["state"] == RunState.RUNNING.value
    assert state_events[-1].payload["state"] == RunState.DONE.value


async def test_tool_call_followed_immediately_by_tool_result():
    provider = ScriptedProvider(
        [
            ScriptedTurn(tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})]),
            ScriptedTurn(text="done after tool"),
        ]
    )
    engine = Engine(provider, tools=[_echo_tool()], config=EngineConfig(model="fake"))
    session = _session()

    await _collect(engine.run(session))

    roles = [m.role for m in session.messages]
    assert roles == [Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]

    assistant_with_calls = session.messages[0]
    tool_msg = session.messages[1]
    assert len(assistant_with_calls.tool_calls) == 1
    assert len(tool_msg.tool_results) == 1
    assert tool_msg.tool_results[0].call_id == assistant_with_calls.tool_calls[0].id
    assert tool_msg.tool_results[0].output == {"echoed": {"x": 1}}
    assert session.state == RunState.DONE


async def test_multiple_tool_calls_all_resolved_before_next_assistant():
    provider = ScriptedProvider(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"a": 1}),
                    ToolCall(id="c2", name="echo", arguments={"b": 2}),
                ]
            ),
            ScriptedTurn(text="ok"),
        ]
    )
    engine = Engine(provider, tools=[_echo_tool()], config=EngineConfig(model="fake"))
    session = _session()

    await _collect(engine.run(session))

    roles = [m.role for m in session.messages]
    assert roles == [Role.ASSISTANT, Role.TOOL, Role.TOOL, Role.ASSISTANT]
    assert session.messages[1].tool_results[0].call_id == "c1"
    assert session.messages[2].tool_results[0].call_id == "c2"


async def test_cancel_interrupts_llm_call_promptly():
    provider = ScriptedProvider([ScriptedTurn(text="slow", delay=5.0)])
    engine = Engine(provider, tools=[], config=EngineConfig(model="fake"))
    session = _session()

    task = asyncio.create_task(_collect(engine.run(session)))
    await asyncio.sleep(0.05)
    engine.cancel()
    events = await asyncio.wait_for(task, timeout=1.0)

    assert session.state == RunState.WAITING_USER
    assert any(e.type == EventType.CANCELLED for e in events)


async def test_unknown_tool_returns_error_result_and_loop_continues():
    provider = ScriptedProvider(
        [
            ScriptedTurn(tool_calls=[ToolCall(id="c1", name="missing", arguments={})]),
            ScriptedTurn(text="handled"),
        ]
    )
    engine = Engine(provider, tools=[], config=EngineConfig(model="fake"))
    session = _session()

    await _collect(engine.run(session))

    tool_msg = session.messages[1]
    assert tool_msg.tool_results[0].is_error is True
    assert "not registered" in tool_msg.tool_results[0].output
    assert session.messages[-1].role == Role.ASSISTANT
    assert session.state == RunState.DONE


async def test_max_steps_terminates_with_done_state():
    provider = ScriptedProvider(
        [ScriptedTurn(tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={})]) for i in range(10)]
    )
    engine = Engine(
        provider,
        tools=[_echo_tool()],
        config=EngineConfig(model="fake", max_steps=3),
    )
    session = _session()

    await _collect(engine.run(session))

    assert session.state == RunState.DONE
    assistant_turns = [m for m in session.messages if m.role == Role.ASSISTANT]
    assert len(assistant_turns) == 3


async def test_engine_passes_provider_options_through_llm_request():
    provider = ScriptedProvider([ScriptedTurn(text="hello")])
    engine = Engine(
        provider,
        tools=[],
        config=EngineConfig(
            model="fake",
            provider_options={"openai": {"reasoning_effort": "high"}},
        ),
    )
    session = _session()

    await _collect(engine.run(session))

    assert len(provider.seen_requests) == 1
    request = provider.seen_requests[0]
    assert request.model == "fake"
    assert request.provider_options == {"openai": {"reasoning_effort": "high"}}


async def test_engine_persists_response_metadata_on_assistant_message():
    provider = ScriptedProvider(
        [
            ScriptedTurn(
                text="hello",
                response_metadata=ProviderResponseMetadata(
                    provider="anthropic",
                    assistant_blocks=[{"type": "text", "text": "hello"}],
                ),
            )
        ]
    )
    engine = Engine(provider, tools=[], config=EngineConfig(model="fake"))
    session = _session()

    await _collect(engine.run(session))

    assert session.messages[-1].extra == {
        LLM_RESPONSE_EXTRA_KEY: {
            "provider": "anthropic",
            "assistant_blocks": [{"type": "text", "text": "hello"}],
        }
    }
    assert get_response_metadata(session.messages[-1].extra) == ProviderResponseMetadata(
        provider="anthropic",
        assistant_blocks=[{"type": "text", "text": "hello"}],
    )
