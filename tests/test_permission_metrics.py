from __future__ import annotations

import pytest

from topsport_agent.engine.permission.metrics import PermissionMetrics
from topsport_agent.types.events import Event, EventType


@pytest.mark.asyncio
async def test_tool_call_end_increments_by_tool_and_outcome():
    m = PermissionMetrics()
    await m.on_event(Event(
        type=EventType.TOOL_CALL_END,
        session_id="s",
        payload={"name": "read_file", "call_id": "c1", "is_error": False},
    ))
    await m.on_event(Event(
        type=EventType.TOOL_CALL_END,
        session_id="s",
        payload={"name": "read_file", "call_id": "c2", "is_error": False},
    ))
    await m.on_event(Event(
        type=EventType.TOOL_CALL_END,
        session_id="s",
        payload={"name": "bash", "call_id": "c3", "is_error": True},
    ))
    snap = m.snapshot()
    assert snap["tool_calls"][("read_file", "allowed")] == 2
    assert snap["tool_calls"][("bash", "error")] == 1


@pytest.mark.asyncio
async def test_unrelated_events_ignored():
    m = PermissionMetrics()
    await m.on_event(Event(
        type=EventType.LLM_CALL_END, session_id="s",
        payload={"step": 0, "tool_call_count": 0, "finish_reason": "end", "usage": {}},
    ))
    assert m.snapshot()["tool_calls"] == {}
