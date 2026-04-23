"""ToolSpec.from_model 的 pydantic 驱动 input_schema 测试。

对标 claude-code 的 Tool.inputSchema (Zod) + z.infer<Input> 类型推断。
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


class _Provider:
    name = "p"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._r = list(responses)
        self._i = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        r = self._r[self._i]
        self._i += 1
        return r


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


class _SearchInput(BaseModel):
    query: str = Field(description="search term", min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


def test_from_model_auto_exports_parameters_schema():
    async def handler(inp: _SearchInput, ctx: ToolContext) -> str:
        del ctx
        return f"q={inp.query};lim={inp.limit}"

    spec = ToolSpec.from_model(
        name="search",
        description="Search the web",
        input_model=_SearchInput,
        handler=handler,
    )
    assert spec.input_schema is _SearchInput
    schema = spec.parameters
    # pydantic-generated JSON Schema 应包含两个字段 + description
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert "limit" in schema["properties"]
    assert schema["properties"]["query"]["description"] == "search term"


def test_from_model_preserves_metadata_kwargs():
    """from_model 要能把 read_only / concurrency_safe 等元数据传进去。"""
    async def handler(inp: _SearchInput, ctx: ToolContext) -> str:
        del ctx
        return inp.query

    spec = ToolSpec.from_model(
        name="search",
        description="d",
        input_model=_SearchInput,
        handler=handler,
        read_only=True,
        concurrency_safe=True,
        trust_level="untrusted",
        max_result_chars=5000,
    )
    assert spec.read_only is True
    assert spec.concurrency_safe is True
    assert spec.trust_level == "untrusted"
    assert spec.max_result_chars == 5000


# ---------------------------------------------------------------------------
# Integration: handler receives typed model, validation errors returned to LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_receives_typed_pydantic_instance():
    received: list[object] = []

    async def handler(inp: _SearchInput, ctx: ToolContext) -> str:
        del ctx
        received.append(inp)
        return f"ok:{inp.query}"

    spec = ToolSpec.from_model(
        name="search", description="d",
        input_model=_SearchInput, handler=handler,
    )
    provider = _Provider([
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "python", "limit": 5})],
            finish_reason="tool_use", usage={}, response_metadata=None,
        ),
        LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                    usage={}, response_metadata=None),
    ])
    engine = Engine(provider, [spec], EngineConfig(model="m"))
    session = Session(id="s", system_prompt="t")
    session.messages.append(Message(role=Role.USER, content="go"))
    async for _ in engine.run(session):
        pass

    assert len(received) == 1
    inp = received[0]
    assert isinstance(inp, _SearchInput)
    assert inp.query == "python"
    assert inp.limit == 5


@pytest.mark.asyncio
async def test_validation_error_returned_as_tool_error():
    """参数不符合 schema（缺 query / limit 越界）→ handler 不调用，LLM 收到 detail。"""
    called = False

    async def handler(inp: _SearchInput, ctx: ToolContext) -> str:
        nonlocal called
        called = True
        del ctx
        return "should not reach"

    spec = ToolSpec.from_model(
        name="search", description="d",
        input_model=_SearchInput, handler=handler,
    )
    provider = _Provider([
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name="search", arguments={"limit": 999})],  # no query, limit oob
            finish_reason="tool_use", usage={}, response_metadata=None,
        ),
        LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                    usage={}, response_metadata=None),
    ])
    engine = Engine(provider, [spec], EngineConfig(model="m"))
    session = Session(id="s", system_prompt="t")
    session.messages.append(Message(role=Role.USER, content="go"))
    async for _ in engine.run(session):
        pass

    assert called is False
    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    out = tool_msg.tool_results[0].output
    assert isinstance(out, dict)
    assert out["error"] == "invalid_input"
    # pydantic 至少会报 query missing 和 limit too large
    assert len(out["detail"]) >= 1


@pytest.mark.asyncio
async def test_defaults_applied_when_optional_field_missing():
    """limit 有 default=10，LLM 不传 → handler 收到 limit=10。"""
    received: list[int] = []

    async def handler(inp: _SearchInput, ctx: ToolContext) -> str:
        del ctx
        received.append(inp.limit)
        return "ok"

    spec = ToolSpec.from_model(
        name="search", description="d",
        input_model=_SearchInput, handler=handler,
    )
    provider = _Provider([
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "go"})],  # no limit
            finish_reason="tool_use", usage={}, response_metadata=None,
        ),
        LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                    usage={}, response_metadata=None),
    ])
    engine = Engine(provider, [spec], EngineConfig(model="m"))
    session = Session(id="s", system_prompt="t")
    session.messages.append(Message(role=Role.USER, content="go"))
    async for _ in engine.run(session):
        pass

    assert received == [10]
