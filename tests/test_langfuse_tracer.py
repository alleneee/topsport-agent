from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.observability import LangfuseTracer
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.message import ToolCall
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


@dataclass
class MockSpan:
    name: str
    as_type: str
    input: Any = None
    model: str | None = None
    parent: MockSpan | MockClient | None = None
    children: list[MockSpan] = field(default_factory=list)
    updates: list[dict[str, Any]] = field(default_factory=list)
    trace_updates: list[dict[str, Any]] = field(default_factory=list)
    ended: bool = False

    def start_observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Any = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MockSpan:
        child = MockSpan(
            name=name, as_type=as_type, input=input, model=model, parent=self
        )
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def update_trace(self, **kwargs: Any) -> None:
        self.trace_updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


@dataclass
class MockClient:
    root_spans: list[MockSpan] = field(default_factory=list)
    flushed: int = 0

    def start_observation(
        self,
        *,
        name: str,
        as_type: str = "span",
        input: Any = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MockSpan:
        span = MockSpan(name=name, as_type=as_type, input=input, model=model, parent=self)
        self.root_spans.append(span)
        return span

    def flush(self) -> None:
        self.flushed += 1


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
    return Session(id="sess-trace", system_prompt="sys", goal="refactor thing")


async def _collect(agen):
    return [event async for event in agen]


async def test_langfuse_tracer_run_start_creates_root_observation():
    client = MockClient()
    tracer = LangfuseTracer(client=client)
    await tracer.on_event(
        Event(
            type=EventType.RUN_START,
            session_id="s1",
            payload={"model": "fake", "goal": "do it"},
        )
    )

    assert len(client.root_spans) == 1
    root = client.root_spans[0]
    assert root.as_type == "agent"
    assert "agent.run[s1]" in root.name
    assert root.input == {"model": "fake", "goal": "do it"}
    assert any(u.get("session_id") == "s1" for u in root.trace_updates)


async def test_langfuse_tracer_end_to_end_through_engine():
    client = MockClient()
    tracer = LangfuseTracer(client=client)

    provider = CapturingProvider(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})],
                finish_reason="tool_use",
                usage={"input_tokens": 10, "output_tokens": 5},
            ),
            LLMResponse(text="final", finish_reason="stop"),
        ]
    )
    engine = Engine(
        provider,
        tools=[_echo_tool()],
        config=EngineConfig(model="gpt-4o"),
        event_subscribers=[tracer],
    )
    session = _session()

    await _collect(engine.run(session))

    assert len(client.root_spans) == 1
    root = client.root_spans[0]
    assert root.ended is True
    assert root.as_type == "agent"

    step_spans = [c for c in root.children if c.as_type == "span"]
    assert len(step_spans) == 2

    first_step = step_spans[0]
    llm_children = [c for c in first_step.children if c.as_type == "generation"]
    assert len(llm_children) == 1
    assert llm_children[0].name == "llm.call"
    assert llm_children[0].model == "gpt-4o"
    assert llm_children[0].ended is True

    llm_end_update = llm_children[0].updates[0]
    assert llm_end_update["usage_details"] == {"input_tokens": 10, "output_tokens": 5}

    tool_children = [c for c in first_step.children if c.as_type == "tool"]
    assert len(tool_children) == 1
    assert tool_children[0].name == "tool.echo"
    assert tool_children[0].ended is True

    assert client.flushed == 1


async def test_langfuse_tracer_tool_error_marks_span_level():
    client = MockClient()
    tracer = LangfuseTracer(client=client)

    await tracer.on_event(
        Event(type=EventType.RUN_START, session_id="s1", payload={"model": "fake"})
    )
    await tracer.on_event(
        Event(type=EventType.STEP_START, session_id="s1", payload={"step": 0})
    )
    await tracer.on_event(
        Event(
            type=EventType.TOOL_CALL_START,
            session_id="s1",
            payload={"name": "broken", "call_id": "c1", "registered": True},
        )
    )
    await tracer.on_event(
        Event(
            type=EventType.TOOL_CALL_END,
            session_id="s1",
            payload={"name": "broken", "call_id": "c1", "is_error": True},
        )
    )

    root = client.root_spans[0]
    step = root.children[0]
    tool_span = step.children[0]
    assert tool_span.as_type == "tool"
    last_update = tool_span.updates[-1]
    assert last_update["level"] == "ERROR"
    assert tool_span.ended is True


async def test_langfuse_tracer_error_event_updates_root_level():
    client = MockClient()
    tracer = LangfuseTracer(client=client)

    await tracer.on_event(
        Event(type=EventType.RUN_START, session_id="s1", payload={"model": "fake"})
    )
    await tracer.on_event(
        Event(
            type=EventType.ERROR,
            session_id="s1",
            payload={"kind": "RuntimeError", "message": "kaboom"},
        )
    )

    root = client.root_spans[0]
    assert any(u.get("level") == "ERROR" for u in root.updates)
    assert any(u.get("status_message") == "kaboom" for u in root.updates)


async def test_langfuse_tracer_handles_client_exceptions_silently():
    class FaultyClient:
        def start_observation(self, **kwargs):
            raise RuntimeError("client down")

        def flush(self):
            raise RuntimeError("flush down")

    tracer = LangfuseTracer(client=FaultyClient())
    await tracer.on_event(
        Event(type=EventType.RUN_START, session_id="s1", payload={})
    )
    await tracer.on_event(
        Event(type=EventType.RUN_END, session_id="s1", payload={})
    )


# ---------------------------------------------------------------------------
# H-S2 · 脱敏 + base_url 白名单
# ---------------------------------------------------------------------------


def test_simple_redactor_masks_sensitive_keys() -> None:
    from topsport_agent.observability import SimpleRedactor

    r = SimpleRedactor()
    out = r({"model": "claude", "api_key": "sk-abc", "messages": [{"role": "user", "content": "hi"}]})
    assert out["model"] == "claude"
    assert out["api_key"] == "[REDACTED]"
    assert out["messages"][0]["content"] == "hi"


def test_simple_redactor_masks_sk_ant_pattern() -> None:
    from topsport_agent.observability import SimpleRedactor

    r = SimpleRedactor()
    out = r({"note": "my key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz and works"})
    assert out["note"] == "[REDACTED]"


def test_simple_redactor_preserves_structure() -> None:
    from topsport_agent.observability import SimpleRedactor

    r = SimpleRedactor()
    payload = {"x": [{"token": "t", "value": "v"}], "y": ("a", "b")}
    out = r(payload)
    assert out["x"][0]["token"] == "[REDACTED]"
    assert out["x"][0]["value"] == "v"
    assert out["y"] == ("a", "b")
    # 原 payload 未被修改
    assert payload["x"][0]["token"] == "t"


async def test_langfuse_tracer_redacts_payload_before_send() -> None:
    from topsport_agent.observability import LangfuseTracer, SimpleRedactor

    client = MockClient()
    tracer = LangfuseTracer(client=client, redactor=SimpleRedactor())

    from topsport_agent.types.events import Event, EventType

    await tracer.on_event(
        Event(
            type=EventType.RUN_START,
            session_id="s",
            payload={"api_key": "sk-secret", "model": "m"},
        )
    )

    # root span 的 input 应已脱敏
    assert client.root_spans, "no root span created"
    obs_input = client.root_spans[0].input
    assert obs_input["api_key"] == "[REDACTED]"
    assert obs_input["model"] == "m"


def test_validate_base_url_allowlist() -> None:
    from topsport_agent.observability import validate_base_url
    import pytest as _pytest

    # 空允许列表 → 跳过
    validate_base_url("https://anywhere.example", [])

    # 匹配 → 通过
    validate_base_url("https://cloud.langfuse.com", ["https://cloud.langfuse.com"])

    # 前缀匹配 → 通过
    validate_base_url("https://cloud.langfuse.com/api/v1", ["https://cloud.langfuse.com"])

    # 不匹配 → 抛
    with _pytest.raises(ValueError, match="not in allowlist"):
        validate_base_url("https://evil.example", ["https://cloud.langfuse.com"])
