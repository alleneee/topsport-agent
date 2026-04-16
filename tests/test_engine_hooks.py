from __future__ import annotations

from typing import Any

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.types.events import EventType
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


class CapturingProvider:
    name = "capturing"

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.seen_call_messages: list[list[Message]] = []
        self.seen_tools: list[list[ToolSpec]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.seen_call_messages.append(list(request.messages))
        self.seen_tools.append(list(request.tools))
        if self._index >= len(self._turns):
            return LLMResponse(text="fallback", finish_reason="stop")
        turn = self._turns[self._index]
        self._index += 1
        return turn


class StaticContextProvider:
    name = "static"

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    async def provide(self, session: Session) -> list[Message]:
        self.calls += 1
        return [Message(role=Role.SYSTEM, content=self._content)]


class CountingToolSource:
    name = "counting"

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = tools
        self.calls = 0

    async def list_tools(self) -> list[ToolSpec]:
        self.calls += 1
        return list(self._tools)


class RecordingHook:
    name = "recorder"

    def __init__(self) -> None:
        self.seen_steps: list[int] = []

    async def after_step(self, session: Session, step: int) -> None:
        self.seen_steps.append(step)


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
    return {"got": args}


def _echo_tool(name: str = "echo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object"},
        handler=_echo_handler,
    )


def _session() -> Session:
    return Session(id="hooks-sess", system_prompt="you are a test agent")


async def _collect(agen):
    return [event async for event in agen]


async def test_context_provider_injected_but_not_persisted():
    provider = CapturingProvider([LLMResponse(text="ok", finish_reason="stop")])
    injector = StaticContextProvider("working memory: goal=x")
    engine = Engine(
        provider,
        tools=[],
        config=EngineConfig(model="fake"),
        context_providers=[injector],
    )
    session = _session()

    events = await _collect(engine.run(session))

    assert injector.calls == 1
    assert len(session.messages) == 1
    assert session.messages[0].role == Role.ASSISTANT
    assert all(m.role != Role.SYSTEM for m in session.messages)

    first_call = provider.seen_call_messages[0]
    assert first_call[0].role == Role.SYSTEM
    system_content = first_call[0].content or ""
    assert "you are a test agent" in system_content
    assert "working memory: goal=x" in system_content

    llm_start = next(e for e in events if e.type == EventType.LLM_CALL_START)
    assert llm_start.payload["ephemeral_msg_count"] == 1


async def test_tool_source_merges_into_snapshot_and_fires_tool():
    dynamic_source = CountingToolSource([_echo_tool("from_source")])
    provider = CapturingProvider(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="c1", name="from_source", arguments={"a": 1})],
                finish_reason="tool_use",
            ),
            LLMResponse(text="done", finish_reason="stop"),
        ]
    )
    engine = Engine(
        provider,
        tools=[],
        config=EngineConfig(model="fake"),
        tool_sources=[dynamic_source],
    )
    session = _session()

    await _collect(engine.run(session))

    assert dynamic_source.calls >= 1
    tool_result_msg = session.messages[1]
    assert tool_result_msg.role == Role.TOOL
    assert tool_result_msg.tool_results[0].output == {"got": {"a": 1}}
    assert not tool_result_msg.tool_results[0].is_error
    assert session.state == RunState.DONE

    first_call_tools = provider.seen_tools[0]
    assert any(t.name == "from_source" for t in first_call_tools)


async def test_post_step_hook_fires_after_each_step_including_final():
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
    hook = RecordingHook()
    engine = Engine(
        provider,
        tools=[_echo_tool("echo")],
        config=EngineConfig(model="fake"),
        post_step_hooks=[hook],
    )
    session = _session()

    await _collect(engine.run(session))

    assert hook.seen_steps == [0, 1]


async def test_tool_source_dedup_prefers_builtin():
    dynamic_source = CountingToolSource([_echo_tool("echo")])
    provider = CapturingProvider([LLMResponse(text="ok", finish_reason="stop")])
    builtin = _echo_tool("echo")
    engine = Engine(
        provider,
        tools=[builtin],
        config=EngineConfig(model="fake"),
        tool_sources=[dynamic_source],
    )
    session = _session()

    await _collect(engine.run(session))

    seen = provider.seen_tools[0]
    names = [t.name for t in seen]
    assert names.count("echo") == 1
    assert seen[0] is builtin
