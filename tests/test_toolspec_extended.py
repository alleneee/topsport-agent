"""ToolSpec 元数据扩展 + Engine 工具执行增强的集成测试。

覆盖：
- validate_input pre-flight 拒绝时 handler 不调用
- max_result_chars + blob_store 自动 offload
- concurrency_safe 工具并发执行（wall-clock 测不准但 task 创建数量可测）
- TrustLevel 枚举向后兼容字符串
- ToolExecutor.wrap 保留所有元数据字段（历史 bug 回归）
"""

from __future__ import annotations

import asyncio
import time

import pytest

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.tools.executor import ToolExecutor
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec, TrustLevel


# ---------------------------------------------------------------------------
# Fixtures / mocks
# ---------------------------------------------------------------------------


class _InMemoryBlobStore:
    """测试用内存 blob store；Protocol 结构化匹配，不继承 BlobStore。"""

    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}

    def store(self, data: str) -> str:
        blob_id = f"blob://test{len(self.blobs)}"
        self.blobs[blob_id] = data
        return blob_id

    def read(self, blob_id: str) -> str | None:
        return self.blobs.get(blob_id)


class _ScriptedProvider:
    name = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


def _build_session_with_user_msg(content: str = "go") -> Session:
    session = Session(id="s1", system_prompt="test")
    session.messages.append(Message(role=Role.USER, content=content))
    return session


# ---------------------------------------------------------------------------
# Unit: ToolSpec dataclass 字段
# ---------------------------------------------------------------------------


def test_toolspec_metadata_defaults():
    async def noop(args: dict, ctx: ToolContext) -> str:
        return "ok"

    spec = ToolSpec(name="t", description="d", parameters={}, handler=noop)
    assert spec.trust_level == "trusted"
    assert spec.read_only is False
    assert spec.destructive is False
    assert spec.concurrency_safe is False
    assert spec.max_result_chars is None
    assert spec.validate_input is None


def test_trust_level_enum_equals_string():
    """StrEnum 向后兼容：TrustLevel.UNTRUSTED == 'untrusted'，sanitizer 对比不会炸。"""
    assert TrustLevel.UNTRUSTED == "untrusted"
    assert TrustLevel.TRUSTED == "trusted"


def test_tool_executor_wrap_preserves_all_fields():
    """历史 bug 回归：ToolExecutor.wrap 之前重建 ToolSpec 时丢字段。现在走 dataclasses.replace。"""
    async def noop(args: dict, ctx: ToolContext) -> str:
        return "ok"

    spec = ToolSpec(
        name="t",
        description="d",
        parameters={},
        handler=noop,
        trust_level="untrusted",
        read_only=True,
        destructive=False,
        concurrency_safe=True,
        max_result_chars=500,
    )
    executor = ToolExecutor()
    wrapped = executor.wrap(spec)
    assert wrapped.trust_level == "untrusted"
    assert wrapped.read_only is True
    assert wrapped.concurrency_safe is True
    assert wrapped.max_result_chars == 500


# ---------------------------------------------------------------------------
# Integration: validate_input pre-flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_input_blocks_handler_when_returns_error():
    """validate_input 返回 str → handler 不调用 + 错误返给 LLM；tool_result 标 is_error。"""
    handler_called = False

    async def handler(args: dict, ctx: ToolContext) -> str:
        nonlocal handler_called
        handler_called = True
        return "should not see this"

    async def validator(args: dict) -> str | None:
        if not args.get("text"):
            return "missing required field: text"
        return None

    tool = ToolSpec(
        name="t",
        description="d",
        parameters={"type": "object"},
        handler=handler,
        validate_input=validator,
    )
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="t", arguments={})],  # missing text
                finish_reason="tool_use",
                usage={},
                response_metadata=None,
            ),
            LLMResponse(
                text="done", tool_calls=[], finish_reason="end_turn",
                usage={}, response_metadata=None,
            ),
        ]
    )
    engine = Engine(provider, [tool], EngineConfig(model="m"))
    session = _build_session_with_user_msg()
    async for _ in engine.run(session):
        pass

    assert handler_called is False, "validate_input returning error should skip handler"
    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    assert tool_msg.tool_results[0].is_error is True
    assert "missing required field" in str(tool_msg.tool_results[0].output)


@pytest.mark.asyncio
async def test_validate_input_passes_when_returns_none():
    async def handler(args: dict, ctx: ToolContext) -> str:
        return f"echo:{args['text']}"

    async def validator(args: dict) -> str | None:
        return None  # always pass

    tool = ToolSpec(
        name="t", description="d", parameters={}, handler=handler, validate_input=validator,
    )
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="t", arguments={"text": "hi"})],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]
    )
    engine = Engine(provider, [tool], EngineConfig(model="m"))
    session = _build_session_with_user_msg()
    async for _ in engine.run(session):
        pass

    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    assert tool_msg.tool_results[0].is_error is False
    assert tool_msg.tool_results[0].output == "echo:hi"


@pytest.mark.asyncio
async def test_validate_input_exception_becomes_tool_error():
    """validator 本身抛异常 → 作为 is_error 返回，不崩 engine。"""
    async def handler(args: dict, ctx: ToolContext) -> str:
        return "ok"

    async def bad_validator(args: dict) -> str | None:
        raise RuntimeError("validator broke")

    tool = ToolSpec(
        name="t", description="d", parameters={}, handler=handler, validate_input=bad_validator,
    )
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="t", arguments={})],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]
    )
    engine = Engine(provider, [tool], EngineConfig(model="m"))
    session = _build_session_with_user_msg()
    async for _ in engine.run(session):
        pass

    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    assert tool_msg.tool_results[0].is_error is True
    assert "RuntimeError" in str(tool_msg.tool_results[0].output)


# ---------------------------------------------------------------------------
# Integration: max_result_chars + blob offload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_result_chars_triggers_blob_offload():
    """handler 返回超大字符串 → 自动落盘 + 返给 LLM 的是 {preview, blob_ref, original_size}。"""
    big_payload = "X" * 10_000

    async def handler(args: dict, ctx: ToolContext) -> str:
        return big_payload

    tool = ToolSpec(
        name="big", description="d", parameters={}, handler=handler,
        max_result_chars=500,  # cap 远小于 payload
    )
    blob_store = _InMemoryBlobStore()
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="big", arguments={})],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]
    )
    engine = Engine(
        provider, [tool], EngineConfig(model="m"),
        blob_store=blob_store,
    )
    session = _build_session_with_user_msg()
    async for _ in engine.run(session):
        pass

    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    out = tool_msg.tool_results[0].output
    assert isinstance(out, dict)
    assert out["truncated"] is True
    assert out["original_size"] == 10_000
    assert out["cap"] == 500
    assert out["blob_ref"].startswith("blob://")
    assert len(out["preview"]) == 500
    # 全量 payload 在 blob_store 里可读回
    assert blob_store.read(out["blob_ref"]) == big_payload


@pytest.mark.asyncio
async def test_max_result_chars_under_cap_untouched():
    """小 payload 不触发 blob offload，原样传回给 LLM。"""
    async def handler(args: dict, ctx: ToolContext) -> str:
        return "tiny"

    tool = ToolSpec(
        name="t", description="d", parameters={}, handler=handler,
        max_result_chars=500,
    )
    blob_store = _InMemoryBlobStore()
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="t", arguments={})],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]
    )
    engine = Engine(
        provider, [tool], EngineConfig(model="m"),
        blob_store=blob_store,
    )
    session = _build_session_with_user_msg()
    async for _ in engine.run(session):
        pass

    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    assert tool_msg.tool_results[0].output == "tiny"
    assert len(blob_store.blobs) == 0


@pytest.mark.asyncio
async def test_default_max_result_chars_applied_when_tool_unset():
    """ToolSpec 没设 max_result_chars → 走 Engine.default_max_result_chars。"""
    async def handler(args: dict, ctx: ToolContext) -> str:
        return "Y" * 3_000

    tool = ToolSpec(name="t", description="d", parameters={}, handler=handler)
    blob_store = _InMemoryBlobStore()
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="t", arguments={})],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]
    )
    engine = Engine(
        provider, [tool], EngineConfig(model="m"),
        blob_store=blob_store,
        default_max_result_chars=1000,
    )
    session = _build_session_with_user_msg()
    async for _ in engine.run(session):
        pass

    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    out = tool_msg.tool_results[0].output
    assert isinstance(out, dict) and out["truncated"] is True
    assert out["cap"] == 1000


# ---------------------------------------------------------------------------
# Integration: concurrency_safe 并发执行
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_safe_tools_run_in_parallel():
    """3 个 concurrency_safe 工具、每个 sleep 50ms → 并发总耗时远小于 3*50=150ms。"""
    async def slow_handler(args: dict, ctx: ToolContext) -> str:
        await asyncio.sleep(0.05)
        return f"done:{args.get('id')}"

    tool = ToolSpec(
        name="slow", description="d", parameters={}, handler=slow_handler,
        concurrency_safe=True,
    )
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="c1", name="slow", arguments={"id": 1}),
                    ToolCall(id="c2", name="slow", arguments={"id": 2}),
                    ToolCall(id="c3", name="slow", arguments={"id": 3}),
                ],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]
    )
    engine = Engine(provider, [tool], EngineConfig(model="m"))
    session = _build_session_with_user_msg()

    start = time.monotonic()
    async for _ in engine.run(session):
        pass
    elapsed = time.monotonic() - start

    # 并发应远小于串行 3*50ms；放宽到 120ms 避免 CI 抖动。
    assert elapsed < 0.12, f"expected concurrent execution, got {elapsed:.3f}s"

    # 事件顺序仍按 calls 列表顺序
    tool_results = [
        m.tool_results[0] for m in session.messages if m.role == Role.TOOL
    ]
    assert [r.output for r in tool_results] == ["done:1", "done:2", "done:3"]


@pytest.mark.asyncio
async def test_unsafe_tools_serialized():
    """没标 concurrency_safe 的工具仍然串行 → 总时长 >= N * 单次。"""
    async def slow_handler(args: dict, ctx: ToolContext) -> str:
        await asyncio.sleep(0.03)
        return "done"

    tool = ToolSpec(
        name="slow", description="d", parameters={}, handler=slow_handler,
        concurrency_safe=False,
    )
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="c1", name="slow", arguments={}),
                    ToolCall(id="c2", name="slow", arguments={}),
                    ToolCall(id="c3", name="slow", arguments={}),
                ],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]
    )
    engine = Engine(provider, [tool], EngineConfig(model="m"))
    session = _build_session_with_user_msg()

    start = time.monotonic()
    async for _ in engine.run(session):
        pass
    elapsed = time.monotonic() - start

    # 3 * 30ms = 90ms 下限
    assert elapsed >= 0.09, f"expected serial execution, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_mixed_safe_and_unsafe_in_same_batch():
    """同批混合：safe 预调度并发，unsafe 走同步路径；最终事件顺序稳定。"""
    order: list[str] = []

    async def safe(args: dict, ctx: ToolContext) -> str:
        order.append(f"safe-start-{args['id']}")
        await asyncio.sleep(0.02)
        order.append(f"safe-end-{args['id']}")
        return f"s:{args['id']}"

    async def unsafe(args: dict, ctx: ToolContext) -> str:
        order.append(f"unsafe-start-{args['id']}")
        await asyncio.sleep(0.02)
        order.append(f"unsafe-end-{args['id']}")
        return f"u:{args['id']}"

    safe_tool = ToolSpec(name="safe", description="", parameters={},
                         handler=safe, concurrency_safe=True)
    unsafe_tool = ToolSpec(name="unsafe", description="", parameters={},
                           handler=unsafe, concurrency_safe=False)
    provider = _ScriptedProvider(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="c1", name="safe", arguments={"id": 1}),
                    ToolCall(id="c2", name="unsafe", arguments={"id": 2}),
                    ToolCall(id="c3", name="safe", arguments={"id": 3}),
                ],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]
    )
    engine = Engine(provider, [safe_tool, unsafe_tool], EngineConfig(model="m"))
    session = _build_session_with_user_msg()
    async for _ in engine.run(session):
        pass

    tool_results = [
        m.tool_results[0] for m in session.messages if m.role == Role.TOOL
    ]
    # 事件顺序按 calls 原序：s:1, u:2, s:3
    assert [r.output for r in tool_results] == ["s:1", "u:2", "s:3"]
    # 两个 safe 应在 unsafe 之前就都 start（预调度后台跑）
    assert order.index("safe-start-1") < order.index("unsafe-start-2")
    assert order.index("safe-start-3") < order.index("unsafe-start-2")
