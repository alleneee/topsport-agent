from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.tools import (
    FileBlobStore,
    ShellInjectionError,
    ToolExecutor,
    enforce_cap,
    safe_exec,
)
from topsport_agent.types.tool import ToolContext, ToolSpec


@pytest.fixture
def blob_dir(tmp_path: Path) -> Path:
    return tmp_path / "blobs"


@pytest.fixture
def cancel_event() -> asyncio.Event:
    return asyncio.Event()


def _ctx(cancel_event: asyncio.Event) -> ToolContext:
    return ToolContext(session_id="s1", call_id="c1", cancel_event=cancel_event)


def test_enforce_cap_under_limit():
    result = enforce_cap("short text", cap=100)
    assert result.truncated is False
    assert result.output == "short text"


def test_enforce_cap_over_limit_no_blob():
    data = "x" * 200
    result = enforce_cap(data, cap=50)
    assert result.truncated is True
    assert result.original_size == 200
    assert result.blob_ref is None
    assert result.output["preview"] == "x" * 50
    assert result.output["cap"] == 50


def test_enforce_cap_over_limit_with_blob(blob_dir: Path):
    store = FileBlobStore(blob_dir)
    data = "y" * 300
    result = enforce_cap(data, cap=50, blob_store=store)
    assert result.truncated is True
    assert result.blob_ref is not None
    assert result.blob_ref.startswith("blob://")
    assert result.output["blob_ref"] == result.blob_ref

    restored = store.read(result.blob_ref)
    assert restored == data


def test_enforce_cap_dict_output():
    data = {"key": "v" * 200}
    result = enforce_cap(data, cap=50)
    assert result.truncated is True
    assert '"key"' in result.output["preview"]


def test_blob_store_read_missing(blob_dir: Path):
    store = FileBlobStore(blob_dir)
    assert store.read("blob://nonexistent") is None


def test_blob_store_roundtrip(blob_dir: Path):
    store = FileBlobStore(blob_dir)
    blob_id = store.store("hello world")
    assert blob_id.startswith("blob://")
    assert store.read(blob_id) == "hello world"


def test_blob_store_deterministic_id(blob_dir: Path):
    store = FileBlobStore(blob_dir)
    id1 = store.store("same content")
    id2 = store.store("same content")
    assert id1 == id2


async def test_tool_executor_wraps_and_truncates(cancel_event: asyncio.Event):
    async def big_handler(args: dict[str, Any], ctx: ToolContext) -> str:
        return "z" * 500

    spec = ToolSpec(
        name="big_tool",
        description="returns large output",
        parameters={"type": "object"},
        handler=big_handler,
    )

    executor = ToolExecutor(caps={"big_tool": 100})
    wrapped = executor.wrap(spec)
    assert wrapped.name == "big_tool"

    output = await wrapped.handler({}, _ctx(cancel_event))
    assert output["truncated"] is True
    assert output["cap"] == 100
    assert len(output["preview"]) == 100


async def test_tool_executor_passes_through_small_output(cancel_event: asyncio.Event):
    async def small_handler(args: dict[str, Any], ctx: ToolContext) -> str:
        return "small"

    spec = ToolSpec(
        name="small_tool",
        description="returns small output",
        parameters={"type": "object"},
        handler=small_handler,
    )

    executor = ToolExecutor(default_cap=1000)
    wrapped = executor.wrap(spec)

    output = await wrapped.handler({}, _ctx(cancel_event))
    assert output == "small"


async def test_tool_executor_wrap_all(cancel_event: asyncio.Event):
    async def handler(args: dict[str, Any], ctx: ToolContext) -> str:
        return "data"

    specs = [
        ToolSpec(name="a", description="a", parameters={}, handler=handler),
        ToolSpec(name="b", description="b", parameters={}, handler=handler),
    ]

    executor = ToolExecutor()
    wrapped = executor.wrap_all(specs)
    assert len(wrapped) == 2
    assert wrapped[0].name == "a"
    assert wrapped[1].name == "b"


async def test_tool_executor_with_blob_offload(
    cancel_event: asyncio.Event, blob_dir: Path
):
    async def handler(args: dict[str, Any], ctx: ToolContext) -> str:
        return "w" * 500

    spec = ToolSpec(
        name="blobbed",
        description="",
        parameters={},
        handler=handler,
    )

    store = FileBlobStore(blob_dir)
    executor = ToolExecutor(caps={"blobbed": 50}, blob_store=store)
    wrapped = executor.wrap(spec)

    output = await wrapped.handler({}, _ctx(cancel_event))
    assert output["truncated"] is True
    assert output["blob_ref"].startswith("blob://")
    assert store.read(output["blob_ref"]) == "w" * 500


async def test_safe_exec_runs_command():
    result = await safe_exec(["echo", "hello"])
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert result["timed_out"] is False


async def test_safe_exec_rejects_string_command():
    with pytest.raises(ShellInjectionError, match="must be a list"):
        await safe_exec("echo hello")  # type: ignore[arg-type]


async def test_safe_exec_rejects_non_string_elements():
    with pytest.raises(ShellInjectionError, match="must be a string"):
        await safe_exec(["echo", 123])  # type: ignore[list-item]


async def test_safe_exec_captures_nonzero_exit():
    result = await safe_exec(["python3", "-c", "import sys; sys.exit(42)"])
    assert result["exit_code"] == 42


async def test_safe_exec_timeout():
    result = await safe_exec(["sleep", "10"], timeout=0.1)
    assert result["timed_out"] is True
    assert result["exit_code"] == -1


async def test_safe_exec_truncates_large_output():
    result = await safe_exec(
        ["python3", "-c", "print('x' * 50000)"],
        max_output=100,
    )
    assert len(result["stdout"]) == 100
    assert result["stdout_truncated"] is True
