"""LLM 流式输出测试：Provider.stream + Engine 的 LLM_TEXT_DELTA + 聚合 response。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from topsport_agent.engine import Engine, EngineConfig
from topsport_agent.llm.provider import StreamingLLMProvider
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.llm.stream import LLMStreamChunk
from topsport_agent.types.events import EventType
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.session import Session

# ---------------------------------------------------------------------------
# Mock streaming provider
# ---------------------------------------------------------------------------


@dataclass
class MockStreamingProvider:
    """可编程流式 mock：yield 预设 chunks，最后 yield done。"""

    name: str = "mock-stream"
    # 每次 stream 调用产出的 deltas + 最终 response
    scenarios: list[tuple[list[str], LLMResponse]] = field(default_factory=list)
    calls: list[LLMRequest] = field(default_factory=list)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # stream-only mock 的 complete 兜底
        self.calls.append(request)
        return LLMResponse(text="noop", tool_calls=[], finish_reason="stop", usage={}, response_metadata=None)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        self.calls.append(request)
        if not self.scenarios:
            yield LLMStreamChunk(type="done", final_response=LLMResponse(
                text="", tool_calls=[], finish_reason="stop", usage={}, response_metadata=None,
            ))
            return
        deltas, final = self.scenarios.pop(0)
        for d in deltas:
            yield LLMStreamChunk(type="text_delta", text_delta=d)
        yield LLMStreamChunk(type="done", final_response=final)


# ---------------------------------------------------------------------------
# 基础流式行为
# ---------------------------------------------------------------------------


def test_streaming_provider_protocol_detection() -> None:
    """isinstance(provider, StreamingLLMProvider) 能识别带 stream 方法的实现。"""
    provider = MockStreamingProvider()
    assert isinstance(provider, StreamingLLMProvider)


async def test_stream_chunks_are_deltas_not_cumulative() -> None:
    """Provider.stream 产出的每个 chunk 是增量，不是累积。"""
    final = LLMResponse(
        text="hello world",
        tool_calls=[],
        finish_reason="stop",
        usage={"input_tokens": 1, "output_tokens": 2},
        response_metadata=None,
    )
    provider = MockStreamingProvider(scenarios=[(["hello ", "world"], final)])

    request = LLMRequest(model="m", messages=[Message(role=Role.USER, content="hi")])
    collected: list[str] = []
    done_final: LLMResponse | None = None

    async for chunk in provider.stream(request):
        if chunk.type == "text_delta" and chunk.text_delta:
            collected.append(chunk.text_delta)
        elif chunk.type == "done":
            done_final = chunk.final_response

    assert collected == ["hello ", "world"]
    assert done_final is not None
    assert done_final.text == "hello world"


# ---------------------------------------------------------------------------
# Engine 集成：流式启用时产出 LLM_TEXT_DELTA 事件
# ---------------------------------------------------------------------------


async def test_engine_emits_text_delta_events_when_streaming() -> None:
    """EngineConfig.stream=True + provider 支持流式时，LLM_TEXT_DELTA 事件应按序 emit。"""
    final = LLMResponse(
        text="ABC",
        tool_calls=[],
        finish_reason="stop",
        usage={"input_tokens": 1, "output_tokens": 3},
        response_metadata=None,
    )
    provider = MockStreamingProvider(scenarios=[(["A", "B", "C"], final)])

    engine = Engine(
        provider=provider,  # type: ignore[arg-type]
        tools=[],
        config=EngineConfig(model="m", max_steps=2, stream=True),
    )
    session = Session(id="s", system_prompt="sys")
    session.messages.append(Message(role=Role.USER, content="go"))

    deltas: list[str] = []
    llm_start_stream_flag: bool | None = None
    async for event in engine.run(session):
        if event.type == EventType.LLM_TEXT_DELTA:
            deltas.append(event.payload.get("delta", ""))
        if event.type == EventType.LLM_CALL_START:
            llm_start_stream_flag = event.payload.get("stream")

    assert deltas == ["A", "B", "C"]
    assert llm_start_stream_flag is True
    # 最终 assistant 消息是聚合后的完整文本
    last = session.messages[-1]
    assert last.role == Role.ASSISTANT
    assert last.content == "ABC"


async def test_engine_falls_back_to_complete_when_stream_disabled() -> None:
    """EngineConfig.stream=False 时不调用 stream()，走 complete() 路径。"""
    provider = MockStreamingProvider()
    engine = Engine(
        provider=provider,  # type: ignore[arg-type]
        tools=[],
        config=EngineConfig(model="m", max_steps=2, stream=False),
    )
    session = Session(id="s", system_prompt="")
    session.messages.append(Message(role=Role.USER, content="go"))

    deltas: list[str] = []
    async for event in engine.run(session):
        if event.type == EventType.LLM_TEXT_DELTA:
            deltas.append(event.payload.get("delta", ""))

    # complete() mock 返回 "noop"，不产生 deltas
    assert deltas == []


async def test_engine_stream_handles_tool_calls_in_final_response() -> None:
    """流式结束后 final_response 带 tool_calls 时，Engine 正常走工具调用路径。"""
    tool_call = ToolCall(id="c1", name="fake_tool", arguments={"x": 1})
    # 第一轮：流式产出 "thinking..."，final 带 tool_call
    first_final = LLMResponse(
        text="thinking...",
        tool_calls=[tool_call],
        finish_reason="tool_use",
        usage={},
        response_metadata=None,
    )
    # 第二轮：工具结果后，模型生成最终答案
    second_final = LLMResponse(
        text="done",
        tool_calls=[],
        finish_reason="stop",
        usage={},
        response_metadata=None,
    )
    provider = MockStreamingProvider(scenarios=[
        (["thinking", "..."], first_final),
        (["do", "ne"], second_final),
    ])

    from topsport_agent.types.tool import ToolSpec

    async def fake_handler(args: dict[str, Any], ctx: Any) -> dict[str, Any]:
        return {"result": args.get("x", 0) * 2}

    tool = ToolSpec(
        name="fake_tool", description="", parameters={"type": "object", "properties": {}},
        handler=fake_handler,  # type: ignore[arg-type]
    )

    engine = Engine(
        provider=provider,  # type: ignore[arg-type]
        tools=[tool],
        config=EngineConfig(model="m", max_steps=5, stream=True),
    )
    session = Session(id="s", system_prompt="")
    session.messages.append(Message(role=Role.USER, content="go"))

    deltas: list[str] = []
    tool_calls_seen: list[str] = []
    async for event in engine.run(session):
        if event.type == EventType.LLM_TEXT_DELTA:
            deltas.append(event.payload.get("delta", ""))
        if event.type == EventType.TOOL_CALL_START:
            tool_calls_seen.append(event.payload.get("name", ""))

    assert deltas == ["thinking", "...", "do", "ne"]
    assert tool_calls_seen == ["fake_tool"]
    # 最后一条 assistant 是 "done"
    assistants = [m for m in session.messages if m.role == Role.ASSISTANT]
    assert assistants[-1].content == "done"


async def test_engine_stream_skipped_when_provider_does_not_support() -> None:
    """Provider 不实现 StreamingLLMProvider 时即使 stream=True 也回退到 complete。"""

    @dataclass
    class NonStreamingProvider:
        name: str = "non-stream"
        calls: list[LLMRequest] = field(default_factory=list)

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls.append(request)
            return LLMResponse(
                text="hi", tool_calls=[], finish_reason="stop", usage={}, response_metadata=None,
            )

    provider = NonStreamingProvider()
    engine = Engine(
        provider=provider,  # type: ignore[arg-type]
        tools=[],
        config=EngineConfig(model="m", max_steps=2, stream=True),
    )
    session = Session(id="s", system_prompt="")
    session.messages.append(Message(role=Role.USER, content="go"))

    deltas: list[str] = []
    stream_flag: bool | None = None
    async for event in engine.run(session):
        if event.type == EventType.LLM_TEXT_DELTA:
            deltas.append(event.payload.get("delta", ""))
        if event.type == EventType.LLM_CALL_START:
            stream_flag = event.payload.get("stream")

    assert deltas == []
    assert stream_flag is False  # Engine 检测到 provider 不支持流式
    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# LLMStreamChunk 类型
# ---------------------------------------------------------------------------


def test_stream_chunk_shape() -> None:
    c = LLMStreamChunk(type="text_delta", text_delta="hi")
    assert c.type == "text_delta"
    assert c.text_delta == "hi"
    assert c.final_response is None

    final = LLMResponse(text="x", tool_calls=[], finish_reason="stop", usage={}, response_metadata=None)
    d = LLMStreamChunk(type="done", final_response=final)
    assert d.type == "done"
    assert d.final_response is final
