"""MCP `progress` notification consumption tests.

覆盖：
- default_progress_callback 路由 INFO 日志到 topsport_agent.mcp.progress.<name>
- 4 种字段组合（progress only / +total / +message / +both）的格式
- wrap_progress_callback 隔离 sync 异常 + async 异常 + log warning
- wrap_progress_callback 转发 progress/total/message 给原 callback
- MCPClient.set_progress_callback 字段持久
- MCPClient.call_tool 在 _progress_callback 设置时自动透传 progress_callback 给 SDK
- MCPClient.call_tool 显式 progress_callback 参数覆盖 client default
- MCPManager.set_progress_callback 批量
- ServerConfig.enable_mcp_progress 从 env 解析
- _build_mcp_manager 在 enable_mcp_progress=True 时挂 default callback
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import pytest

from topsport_agent.mcp import (
    MCPClient,
    MCPManager,
    MCPServerConfig,
    default_progress_callback,
    wrap_progress_callback,
)
from topsport_agent.mcp.types import MCPTransport
from topsport_agent.server.app import _build_mcp_manager
from topsport_agent.server.config import ServerConfig


# ---------------------------------------------------------------------------
# default_progress_callback formatting
# ---------------------------------------------------------------------------


def test_default_progress_emits_info_log_progress_only(caplog) -> None:
    cb = default_progress_callback("c1")
    with caplog.at_level("INFO", logger="topsport_agent.mcp.progress.c1"):
        cb(0.5, None, None)
    rec = next(r for r in caplog.records if "progress=" in r.message)
    assert rec.levelno == logging.INFO
    assert rec.message == "progress=0.5"


def test_default_progress_with_total_only(caplog) -> None:
    cb = default_progress_callback("c1")
    with caplog.at_level("INFO", logger="topsport_agent.mcp.progress.c1"):
        cb(2, 10, None)
    rec = next(r for r in caplog.records if "progress=" in r.message)
    assert rec.message == "progress=2/10"


def test_default_progress_with_message_only(caplog) -> None:
    cb = default_progress_callback("c1")
    with caplog.at_level("INFO", logger="topsport_agent.mcp.progress.c1"):
        cb(0.5, None, "loading shards")
    rec = next(r for r in caplog.records if "progress=" in r.message)
    assert rec.message == "progress=0.5 message=loading shards"


def test_default_progress_with_total_and_message(caplog) -> None:
    cb = default_progress_callback("c1")
    with caplog.at_level("INFO", logger="topsport_agent.mcp.progress.c1"):
        cb(3, 5, "done")
    rec = next(r for r in caplog.records if "progress=" in r.message)
    assert rec.message == "progress=3/5 message=done"


# ---------------------------------------------------------------------------
# wrap_progress_callback exception isolation
# ---------------------------------------------------------------------------


async def test_wrap_isolates_sync_exception(caplog) -> None:
    def boom(_p, _t, _m):
        raise RuntimeError("sync boom")

    wrapped = wrap_progress_callback(boom, client_name="x")
    with caplog.at_level("WARNING", logger="topsport_agent.mcp.progress"):
        await wrapped(0.5, None, None)  # should not raise
    assert any("progress_callback raised" in r.message for r in caplog.records)


def test_wrap_progress_callback_requires_client_name() -> None:
    """client_name 是 keyword-only 必填，避免日志中出现无意义 '?' 占位符。"""
    with pytest.raises(TypeError):
        wrap_progress_callback(lambda p, t, m: None)  # type: ignore[call-arg]


async def test_wrap_isolates_async_exception(caplog) -> None:
    async def boom(_p, _t, _m):
        raise RuntimeError("async boom")

    wrapped = wrap_progress_callback(boom, client_name="x")
    with caplog.at_level("WARNING", logger="topsport_agent.mcp.progress"):
        await wrapped(0.5, None, None)
    assert any("progress_callback raised" in r.message for r in caplog.records)


async def test_wrap_forwards_args_to_callback() -> None:
    seen: list[Any] = []

    def cb(progress, total, message):
        seen.append((progress, total, message))

    wrapped = wrap_progress_callback(cb, client_name="x")
    await wrapped(0.5, 1.0, "hi")
    assert seen == [(0.5, 1.0, "hi")]


async def test_wrap_handles_async_callback() -> None:
    seen: list[Any] = []

    async def cb(progress, total, message):
        seen.append((progress, total, message))

    wrapped = wrap_progress_callback(cb, client_name="x")
    await wrapped(2, 10, None)
    assert seen == [(2, 10, None)]


# ---------------------------------------------------------------------------
# MCPClient field + call_tool integration
# ---------------------------------------------------------------------------


def _dummy_factory() -> Any:
    @contextlib.asynccontextmanager
    async def factory():
        yield None

    return factory


async def test_client_default_no_progress_callback() -> None:
    client = MCPClient("s", _dummy_factory())
    assert client.progress_callback is None


async def test_client_set_progress_callback_persists() -> None:
    client = MCPClient("s", _dummy_factory())
    cb = default_progress_callback("s")
    client.set_progress_callback(cb)
    assert client.progress_callback is cb
    client.set_progress_callback(None)
    assert client.progress_callback is None


async def test_call_tool_passes_wrapped_progress_callback_when_set() -> None:
    captured: dict[str, Any] = {}

    class _Sess:
        async def call_tool(self, name, **kwargs):
            captured["call_kwargs"] = kwargs

            class _R:
                content = []
                isError = False
                structuredContent = None

            return _R()

    @contextlib.asynccontextmanager
    async def factory():
        yield _Sess()

    client = MCPClient("s", factory)
    client.set_progress_callback(default_progress_callback("s"))
    await client.call_tool("foo", {"x": 1})
    assert "progress_callback" in captured["call_kwargs"]


async def test_call_tool_omits_progress_callback_when_unset() -> None:
    captured: dict[str, Any] = {}

    class _Sess:
        async def call_tool(self, name, **kwargs):
            captured["call_kwargs"] = kwargs

            class _R:
                content = []
                isError = False
                structuredContent = None

            return _R()

    @contextlib.asynccontextmanager
    async def factory():
        yield _Sess()

    client = MCPClient("s", factory)
    await client.call_tool("foo", {"x": 1})
    assert "progress_callback" not in captured["call_kwargs"]


async def test_call_tool_explicit_progress_overrides_client_default() -> None:
    captured: dict[str, Any] = {}
    seen: list[tuple] = []

    def explicit_cb(p, t, m):
        seen.append(("explicit", p, t, m))

    class _Sess:
        async def call_tool(self, name, **kwargs):
            captured["progress_callback"] = kwargs.get("progress_callback")

            class _R:
                content = []
                isError = False
                structuredContent = None

            return _R()

    @contextlib.asynccontextmanager
    async def factory():
        yield _Sess()

    client = MCPClient("s", factory)
    # Set a default that we expect NOT to be used
    client.set_progress_callback(lambda p, t, m: seen.append(("default", p, t, m)))
    await client.call_tool("foo", {}, progress_callback=explicit_cb)

    # 透传给 SDK 的是 wrap 过的版本；调它一次验证 wrap 转发到 explicit_cb
    wrapped = captured["progress_callback"]
    assert wrapped is not None
    await wrapped(0.7, 1.0, "ok")
    assert seen == [("explicit", 0.7, 1.0, "ok")], (
        "explicit progress_callback 应覆盖 client default"
    )


# ---------------------------------------------------------------------------
# Manager batch
# ---------------------------------------------------------------------------


async def test_manager_set_progress_callback_applies_to_all() -> None:
    manager = MCPManager()
    c1 = MCPClient("a", _dummy_factory())
    c2 = MCPClient("b", _dummy_factory())
    manager.register(c1)
    manager.register(c2)

    cb = default_progress_callback("shared")
    manager.set_progress_callback(cb)

    assert c1.progress_callback is cb
    assert c2.progress_callback is cb

    manager.set_progress_callback(None)
    assert c1.progress_callback is None
    assert c2.progress_callback is None


# ---------------------------------------------------------------------------
# ServerConfig env wiring + _build_mcp_manager
# ---------------------------------------------------------------------------


def test_server_config_default_disables_progress() -> None:
    cfg = ServerConfig()
    assert cfg.enable_mcp_progress is False


def test_server_config_from_env_reads_enable_mcp_progress(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MCP_PROGRESS", "true")
    cfg = ServerConfig.from_env()
    assert cfg.enable_mcp_progress is True


def test_build_mcp_manager_attaches_progress_callback_when_enabled() -> None:
    cfg = ServerConfig(
        enable_brave_search=True, brave_api_key="k",
        enable_mcp_progress=True,
    )
    manager = _build_mcp_manager(cfg)
    assert manager is not None
    brave = manager.get("brave-search")
    assert brave is not None
    assert brave.progress_callback is not None


def test_build_mcp_manager_no_progress_when_disabled() -> None:
    cfg = ServerConfig(
        enable_brave_search=True, brave_api_key="k",
        enable_mcp_progress=False,
    )
    manager = _build_mcp_manager(cfg)
    assert manager is not None
    brave = manager.get("brave-search")
    assert brave is not None
    assert brave.progress_callback is None


# ---------------------------------------------------------------------------
# Tri-state semantic of call_tool(progress_callback=...)
# ---------------------------------------------------------------------------


async def test_call_tool_explicit_none_disables_even_if_default_set() -> None:
    """显式 progress_callback=None 应禁用本次 progress，覆盖 client default。
    防止 Phase 4 抓过的 "None 被 or 默认值覆盖" 模式。
    """
    captured: dict[str, Any] = {}

    class _Sess:
        async def call_tool(self, name, **kwargs):
            captured["call_kwargs"] = kwargs

            class _R:
                content = []
                isError = False
                structuredContent = None

            return _R()

    @contextlib.asynccontextmanager
    async def factory():
        yield _Sess()

    client = MCPClient("s", factory)
    client.set_progress_callback(default_progress_callback("s"))

    await client.call_tool("foo", {}, progress_callback=None)

    assert "progress_callback" not in captured["call_kwargs"], (
        "显式 None 应禁用 progress，不走 client default"
    )


# ---------------------------------------------------------------------------
# tool_bridge 集成路径
# ---------------------------------------------------------------------------


async def test_tool_bridge_handler_routes_progress_through_client_default() -> None:
    """MCPToolSource handler 调 client.call_tool 时，client.progress_callback
    应自动透传给 SDK（端到端验证 5.4 与 tool_bridge 集成）。"""
    from topsport_agent.mcp.tool_bridge import MCPToolSource

    captured: dict[str, Any] = {}

    class _Sess:
        async def call_tool(self, name, **kwargs):
            captured["progress_callback"] = kwargs.get("progress_callback")

            class _R:
                content = []
                isError = False
                structuredContent = None

            return _R()

    @contextlib.asynccontextmanager
    async def factory():
        yield _Sess()

    client = MCPClient("brv", factory)
    cb = default_progress_callback("brv")
    client.set_progress_callback(cb)

    source = MCPToolSource(client)
    # 构造一个 ToolSpec 只为拿到 handler；MCPToolSource._wrap 接受 duck-typed input
    class _MockMCPTool:
        name = "search"
        description = ""
        inputSchema = {"type": "object"}

    tool_spec = source._wrap(_MockMCPTool())
    from topsport_agent.types.tool import ToolContext

    await tool_spec.handler(
        {"q": "x"},
        ToolContext(session_id="s", call_id="c", cancel_event=asyncio.Event()),
    )
    assert captured["progress_callback"] is not None


# ---------------------------------------------------------------------------
# default_progress_callback level + sample_every
# ---------------------------------------------------------------------------


def test_default_progress_uses_custom_level(caplog) -> None:
    cb = default_progress_callback("c1", level=logging.WARNING)
    with caplog.at_level("DEBUG", logger="topsport_agent.mcp.progress.c1"):
        cb(0.1, None, None)
    rec = next(r for r in caplog.records if "progress=" in r.message)
    assert rec.levelno == logging.WARNING


def test_default_progress_sample_every_drops_intermediate(caplog) -> None:
    cb = default_progress_callback("c1", sample_every=3)
    with caplog.at_level("INFO", logger="topsport_agent.mcp.progress.c1"):
        for i in range(7):
            cb(float(i), None, None)
    # sample_every=3 触发：第 1, 4, 7 次 → 3 条记录
    msgs = [r.message for r in caplog.records if "progress=" in r.message]
    assert len(msgs) == 3
    assert msgs == ["progress=0.0", "progress=3.0", "progress=6.0"]


def test_default_progress_sample_every_zero_or_none_no_sampling(caplog) -> None:
    cb = default_progress_callback("c1", sample_every=None)
    with caplog.at_level("INFO", logger="topsport_agent.mcp.progress.c1"):
        for i in range(3):
            cb(float(i), None, None)
    assert len([r for r in caplog.records if "progress=" in r.message]) == 3


# ---------------------------------------------------------------------------
# Logger name with special chars (dot, dash) — naming hygiene
# ---------------------------------------------------------------------------


def test_default_progress_logger_name_with_dotted_client_name(caplog) -> None:
    """client_name 含 . 时 stdlib logging 把它视为父子层级；docstring 已说明
    操作者要避开 — 这条测试钉住"behavior is what it is"。"""
    cb = default_progress_callback("a.b")
    with caplog.at_level("INFO", logger="topsport_agent.mcp.progress.a.b"):
        cb(0.1, None, None)
    assert any(
        r.name == "topsport_agent.mcp.progress.a.b" for r in caplog.records
    )
