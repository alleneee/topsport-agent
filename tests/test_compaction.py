from __future__ import annotations

from typing import Any

from topsport_agent.engine.compaction import (
    CompactionHook,
    auto_compact,
    estimate_tokens,
    micro_compact,
)
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.message import Message, Role, ToolCall, ToolResult
from topsport_agent.types.session import Session


def _msg(role: Role, content: str = "") -> Message:
    return Message(role=role, content=content)


def _tool_msg(call_id: str, output: str) -> Message:
    return Message(
        role=Role.TOOL,
        tool_results=[ToolResult(call_id=call_id, output=output)],
    )


def _assistant_with_call(call_id: str, name: str = "t") -> Message:
    return Message(
        role=Role.ASSISTANT,
        tool_calls=[ToolCall(id=call_id, name=name, arguments={})],
    )


class MockSummaryProvider:
    name = "summary-mock"

    def __init__(self, summary_text: str = "This is a summary.") -> None:
        self._summary = summary_text
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(text=self._summary)


def test_estimate_tokens_basic():
    messages = [_msg(Role.USER, "hello world")]
    tokens = estimate_tokens(messages)
    assert tokens == len("hello world") // 4


def test_estimate_tokens_includes_tool_content():
    messages = [
        _msg(Role.USER, "go"),
        _assistant_with_call("c1", "echo"),
        _tool_msg("c1", "x" * 400),
    ]
    tokens = estimate_tokens(messages)
    assert tokens > 100


def test_micro_compact_preserves_recent_tools():
    messages = [
        _msg(Role.USER, "start"),
        _assistant_with_call("c1"),
        _tool_msg("c1", "result-1"),
        _assistant_with_call("c2"),
        _tool_msg("c2", "result-2"),
        _assistant_with_call("c3"),
        _tool_msg("c3", "result-3"),
    ]

    compacted = micro_compact(messages, keep_recent_tools=2)
    tool_messages = [m for m in compacted if m.role == Role.TOOL]

    assert tool_messages[0].tool_results[0].output == "(previous output omitted)"
    assert tool_messages[1].tool_results[0].output == "result-2"
    assert tool_messages[2].tool_results[0].output == "result-3"


def test_micro_compact_no_change_when_few_tools():
    messages = [
        _msg(Role.USER, "start"),
        _assistant_with_call("c1"),
        _tool_msg("c1", "result-1"),
    ]
    compacted = micro_compact(messages, keep_recent_tools=3)
    assert compacted[2].tool_results[0].output == "result-1"


def test_micro_compact_preserves_error_flag():
    messages = [
        _assistant_with_call("c1"),
        Message(
            role=Role.TOOL,
            tool_results=[ToolResult(call_id="c1", output="err", is_error=True)],
        ),
        _assistant_with_call("c2"),
        _tool_msg("c2", "ok"),
    ]
    compacted = micro_compact(messages, keep_recent_tools=1)
    assert compacted[1].tool_results[0].is_error is True
    assert compacted[1].tool_results[0].output == "(previous output omitted)"


async def test_auto_compact_skips_below_threshold():
    messages = [_msg(Role.USER, "short")]
    provider = MockSummaryProvider()

    result, did = await auto_compact(
        messages,
        session_goal="do stuff",
        system_identity="helper",
        provider=provider,
        summary_model="mock",
        context_window=100_000,
        threshold=0.65,
        keep_recent=4,
    )

    assert did is False
    assert len(provider.calls) == 0
    assert result == messages


async def test_auto_compact_triggers_on_large_context():
    big_content = "x" * 300_000
    messages = [
        _msg(Role.USER, big_content),
        _msg(Role.ASSISTANT, "ok"),
        _msg(Role.USER, "more"),
        _msg(Role.ASSISTANT, "sure"),
        _msg(Role.USER, "recent-1"),
        _msg(Role.ASSISTANT, "recent-2"),
    ]
    provider = MockSummaryProvider("Conversation was about x stuff.")

    result, did = await auto_compact(
        messages,
        session_goal="refactor ingest",
        system_identity="python agent",
        provider=provider,
        summary_model="mock",
        context_window=100_000,
        threshold=0.65,
        keep_recent=2,
    )

    assert did is True
    assert len(provider.calls) == 1
    assert len(result) == 3

    reinject = result[0]
    assert reinject.role == Role.SYSTEM
    reinject_content = reinject.content or ""
    assert "Identity" in reinject_content
    assert "python agent" in reinject_content
    assert "Task goal" in reinject_content
    assert "refactor ingest" in reinject_content
    assert "Conversation was about x stuff." in reinject_content

    assert result[1].content == "recent-1"
    assert result[2].content == "recent-2"


async def test_auto_compact_skips_when_too_few_messages():
    big = "y" * 500_000
    messages = [_msg(Role.USER, big)]
    provider = MockSummaryProvider()

    result, did = await auto_compact(
        messages,
        session_goal=None,
        system_identity=None,
        provider=provider,
        summary_model="mock",
        context_window=100_000,
        threshold=0.65,
        keep_recent=4,
    )

    assert did is False


async def test_auto_compact_handles_summary_failure():
    class FailingProvider:
        name = "failing"

        async def complete(self, request: Any) -> Any:
            raise RuntimeError("api down")

    big = "z" * 500_000
    messages = [
        _msg(Role.USER, big),
        _msg(Role.ASSISTANT, "ok"),
        _msg(Role.USER, "recent"),
    ]

    result, did = await auto_compact(
        messages,
        session_goal=None,
        system_identity=None,
        provider=FailingProvider(),
        summary_model="mock",
        context_window=100_000,
        threshold=0.65,
        keep_recent=1,
    )

    assert did is True
    reinject = result[0]
    assert "summary generation failed" in (reinject.content or "")


async def test_compaction_hook_runs_micro_and_auto():
    big = "a" * 400_000
    session = Session(
        id="s1",
        system_prompt="you are agent",
        goal="fix the bug",
    )
    session.messages.extend(
        [
            _msg(Role.USER, big),
            _assistant_with_call("c1"),
            _tool_msg("c1", "old-result"),
            _assistant_with_call("c2"),
            _tool_msg("c2", "also-old"),
            _assistant_with_call("c3"),
            _tool_msg("c3", "recent-tool"),
            _msg(Role.ASSISTANT, "recent-assistant"),
            _msg(Role.USER, "recent-user"),
        ]
    )

    provider = MockSummaryProvider("Bug investigation summary.")
    hook = CompactionHook(
        provider,
        summary_model="mock",
        context_window=100_000,
        threshold=0.65,
        keep_recent_messages=3,
        keep_recent_tool_results=1,
    )

    await hook.after_step(session, step=5)

    assert any("Bug investigation summary." in (m.content or "") for m in session.messages)
    recent = session.messages[-3:]
    assert recent[0].role == Role.TOOL
    assert recent[0].tool_results[0].output == "recent-tool"


async def test_compaction_hook_micro_only_when_below_threshold():
    session = Session(id="s1", system_prompt="sys")
    session.messages.extend(
        [
            _msg(Role.USER, "short"),
            _assistant_with_call("c1"),
            _tool_msg("c1", "old-tool-output"),
            _assistant_with_call("c2"),
            _tool_msg("c2", "newer-tool-output"),
            _msg(Role.ASSISTANT, "ok"),
        ]
    )

    provider = MockSummaryProvider()
    hook = CompactionHook(
        provider,
        summary_model="mock",
        context_window=100_000,
        threshold=0.65,
        keep_recent_tool_results=1,
    )

    await hook.after_step(session, step=0)

    assert len(provider.calls) == 0
    tool_msgs = [m for m in session.messages if m.role == Role.TOOL]
    assert tool_msgs[0].tool_results[0].output == "(previous output omitted)"
    assert tool_msgs[1].tool_results[0].output == "newer-tool-output"
