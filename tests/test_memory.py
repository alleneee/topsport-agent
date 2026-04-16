from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.memory import (
    FileMemoryStore,
    MemoryEntry,
    MemoryInjector,
    MemoryType,
    build_memory_tools,
)
from topsport_agent.types.message import Role
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext


@pytest.fixture
def store(tmp_path: Path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "memory")


@pytest.fixture
def cancel_event() -> asyncio.Event:
    return asyncio.Event()


def _session(session_id: str = "sess-mem") -> Session:
    return Session(id=session_id, system_prompt="sys")


def _ctx(session_id: str, cancel_event: asyncio.Event, call_id: str = "c1") -> ToolContext:
    return ToolContext(session_id=session_id, call_id=call_id, cancel_event=cancel_event)


async def test_file_store_write_read_roundtrip(store: FileMemoryStore):
    entry = MemoryEntry(
        key="task_goal",
        name="Task goal",
        description="What the user wants done",
        type=MemoryType.GOAL,
        content="Refactor ingest pipeline to async generators.",
    )
    await store.write("s1", entry)

    reloaded = await store.read("s1", "task_goal")
    assert reloaded is not None
    assert reloaded.key == "task_goal"
    assert reloaded.name == "Task goal"
    assert reloaded.type == MemoryType.GOAL
    assert reloaded.content == "Refactor ingest pipeline to async generators."
    assert reloaded.created_at is not None
    assert reloaded.updated_at is not None


async def test_file_store_list_sorted_and_session_scoped(store: FileMemoryStore):
    await store.write(
        "s1",
        MemoryEntry(
            key="b_fact",
            name="B",
            description="",
            type=MemoryType.FACT,
            content="b content",
        ),
    )
    await store.write(
        "s1",
        MemoryEntry(
            key="a_goal",
            name="A",
            description="",
            type=MemoryType.GOAL,
            content="a content",
        ),
    )
    await store.write(
        "s2",
        MemoryEntry(
            key="other",
            name="other",
            description="",
            type=MemoryType.NOTE,
            content="other session",
        ),
    )

    s1_entries = await store.list("s1")
    assert [e.key for e in s1_entries] == ["a_goal", "b_fact"]

    s2_entries = await store.list("s2")
    assert [e.key for e in s2_entries] == ["other"]


async def test_file_store_delete(store: FileMemoryStore):
    await store.write(
        "s1",
        MemoryEntry(
            key="temp",
            name="Temp",
            description="",
            type=MemoryType.NOTE,
            content="temporary",
        ),
    )
    assert await store.read("s1", "temp") is not None

    deleted = await store.delete("s1", "temp")
    assert deleted is True
    assert await store.read("s1", "temp") is None

    assert await store.delete("s1", "temp") is False


async def test_file_store_unknown_read_returns_none(store: FileMemoryStore):
    assert await store.read("s1", "nope") is None
    assert await store.list("s1") == []


async def test_memory_injector_produces_system_message(store: FileMemoryStore):
    await store.write(
        "s1",
        MemoryEntry(
            key="task_goal",
            name="Task goal",
            description="What user wants",
            type=MemoryType.GOAL,
            content="Refactor X",
        ),
    )
    await store.write(
        "s1",
        MemoryEntry(
            key="identity",
            name="Who you are",
            description="Runtime identity",
            type=MemoryType.IDENTITY,
            content="You are a Python refactoring agent.",
        ),
    )

    injector = MemoryInjector(store)
    messages = await injector.provide(_session("s1"))

    assert len(messages) == 1
    assert messages[0].role == Role.SYSTEM
    content = messages[0].content or ""
    assert "Working memory" in content
    assert "[goal] Task goal" in content
    assert "Refactor X" in content
    assert "[identity] Who you are" in content
    assert "You are a Python refactoring agent." in content


async def test_memory_injector_empty_store_yields_nothing(store: FileMemoryStore):
    injector = MemoryInjector(store)
    messages = await injector.provide(_session("s1"))
    assert messages == []


async def test_memory_injector_type_filter(store: FileMemoryStore):
    await store.write(
        "s1",
        MemoryEntry(
            key="g",
            name="g",
            description="",
            type=MemoryType.GOAL,
            content="goal content",
        ),
    )
    await store.write(
        "s1",
        MemoryEntry(
            key="n",
            name="n",
            description="",
            type=MemoryType.NOTE,
            content="note content",
        ),
    )

    injector = MemoryInjector(store, types=[MemoryType.GOAL])
    messages = await injector.provide(_session("s1"))

    content = messages[0].content or ""
    assert "goal content" in content
    assert "note content" not in content


async def test_memory_tools_save_recall_forget(
    store: FileMemoryStore, cancel_event: asyncio.Event
):
    tools = {t.name: t for t in build_memory_tools(store)}
    ctx = _ctx("sess-mem", cancel_event)

    save_result: dict[str, Any] = await tools["save_memory"].handler(
        {
            "key": "task_goal",
            "name": "Task goal",
            "description": "Test",
            "type": "goal",
            "content": "Do the thing.",
        },
        ctx,
    )
    assert save_result == {"ok": True, "key": "task_goal", "type": "goal"}

    recall_one: dict[str, Any] = await tools["recall_memory"].handler(
        {"key": "task_goal"}, ctx
    )
    assert recall_one["found"] is True
    assert recall_one["entry"]["type"] == "goal"
    assert recall_one["entry"]["content"] == "Do the thing."

    recall_all: dict[str, Any] = await tools["recall_memory"].handler({}, ctx)
    assert recall_all["count"] == 1
    assert recall_all["entries"][0]["key"] == "task_goal"

    forget_result: dict[str, Any] = await tools["forget_memory"].handler(
        {"key": "task_goal"}, ctx
    )
    assert forget_result == {"deleted": True, "key": "task_goal"}

    recall_gone: dict[str, Any] = await tools["recall_memory"].handler(
        {"key": "task_goal"}, ctx
    )
    assert recall_gone["found"] is False


async def test_memory_tool_invalid_type_rejected(
    store: FileMemoryStore, cancel_event: asyncio.Event
):
    tools = {t.name: t for t in build_memory_tools(store)}
    result: dict[str, Any] = await tools["save_memory"].handler(
        {"key": "x", "type": "not-a-type", "content": "body"},
        _ctx("sess-mem", cancel_event),
    )
    assert result["ok"] is False
    assert "not-a-type" in result["error"]
