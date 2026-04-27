"""MCP `logging` capability tests.

覆盖：
- 8 级 MCP level → stdlib level 映射（含 NOTICE 在 INFO/WARNING 之间）
- 未知级别落回 INFO + 一次性 warn
- default_logging_callback 路由消息到 topsport_agent.mcp.server.<name>.<logger>
- dict data 与 string data 的 dispatch 形态
- MCPClient.set_logging_callback / set_logging_level 字段持久
- 设置 logging_level 后 _make_real_session_factory 调 set_logging_level
- ServerConfig.mcp_log_level 从 env 解析
- _build_mcp_manager 在 mcp_log_level 非空且合法时挂 callback
- 无效 log level 启动期 fail-fast
- "off" / "none" / 空字符串视为关闭
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
    default_logging_callback,
    mcp_level_to_python,
)
from topsport_agent.mcp.types import MCPTransport
from topsport_agent.server.app import _build_mcp_manager
from topsport_agent.server.config import ServerConfig


# ---------------------------------------------------------------------------
# Level mapping
# ---------------------------------------------------------------------------


def test_mcp_level_to_python_maps_all_eight_levels() -> None:
    assert mcp_level_to_python("debug") == logging.DEBUG
    assert mcp_level_to_python("info") == logging.INFO
    assert mcp_level_to_python("notice") == logging.INFO + 5
    assert mcp_level_to_python("warning") == logging.WARNING
    assert mcp_level_to_python("error") == logging.ERROR
    assert mcp_level_to_python("critical") == logging.CRITICAL
    assert mcp_level_to_python("alert") == logging.CRITICAL
    assert mcp_level_to_python("emergency") == logging.CRITICAL


def test_mcp_level_to_python_case_insensitive() -> None:
    assert mcp_level_to_python("WARNING") == logging.WARNING
    assert mcp_level_to_python("Error") == logging.ERROR


def test_mcp_level_to_python_unknown_falls_back_to_info(caplog) -> None:
    with caplog.at_level("WARNING", logger="topsport_agent.mcp.logging_handler"):
        result = mcp_level_to_python("custom-fancy")
    assert result == logging.INFO
    assert any("unknown MCP log level" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# default_logging_callback dispatch
# ---------------------------------------------------------------------------


class _FakeLogParams:
    def __init__(self, level: str, logger: str | None, data: Any) -> None:
        self.level = level
        self.logger = logger
        self.data = data


async def test_default_callback_routes_to_named_logger(caplog) -> None:
    cb = default_logging_callback("test-client")
    with caplog.at_level("DEBUG", logger="topsport_agent.mcp.server.test-client"):
        await cb(_FakeLogParams("info", None, "hello world"))
    assert any("hello world" in rec.message for rec in caplog.records)


async def test_default_callback_appends_server_logger_name(caplog) -> None:
    cb = default_logging_callback("brave-search")
    with caplog.at_level("DEBUG", logger="topsport_agent.mcp.server.brave-search.subsys"):
        await cb(_FakeLogParams("warning", "subsys", "alert message"))
    assert any(
        rec.name == "topsport_agent.mcp.server.brave-search.subsys"
        for rec in caplog.records
    )


async def test_default_callback_dict_data_serialised(caplog) -> None:
    cb = default_logging_callback("c")
    with caplog.at_level("INFO", logger="topsport_agent.mcp.server.c"):
        await cb(_FakeLogParams("info", None, {"key": "value", "n": 42}))
    rec = next(r for r in caplog.records if r.name.startswith("topsport_agent.mcp.server.c"))
    assert "key='value'" in rec.message
    assert getattr(rec, "mcp_data") == {"key": "value", "n": 42}


async def test_default_callback_emits_at_correct_severity(caplog) -> None:
    cb = default_logging_callback("c")
    with caplog.at_level("DEBUG", logger="topsport_agent.mcp.server.c"):
        await cb(_FakeLogParams("error", None, "bad"))
    rec = next(r for r in caplog.records if "bad" in r.message)
    assert rec.levelno == logging.ERROR


# ---------------------------------------------------------------------------
# MCPClient field persistence
# ---------------------------------------------------------------------------


def _dummy_factory() -> Any:
    @contextlib.asynccontextmanager
    async def factory():
        yield None

    return factory


async def test_client_default_no_logging_callback() -> None:
    client = MCPClient("s", _dummy_factory())
    assert client.logging_callback is None
    assert client.logging_level is None


async def test_client_set_logging_callback_persists() -> None:
    client = MCPClient("s", _dummy_factory())
    cb = default_logging_callback("s")
    client.set_logging_callback(cb, level="warning")
    assert client.logging_callback is cb
    assert client.logging_level == "warning"

    client.set_logging_callback(None)
    assert client.logging_callback is None
    # level unchanged after callback=None
    assert client.logging_level == "warning"


async def test_client_set_logging_level_independently() -> None:
    client = MCPClient("s", _dummy_factory())
    client.set_logging_level("debug")
    assert client.logging_level == "debug"
    client.set_logging_level(None)
    assert client.logging_level is None


# ---------------------------------------------------------------------------
# Manager batch
# ---------------------------------------------------------------------------


async def test_manager_set_logging_callback_applies_to_all() -> None:
    manager = MCPManager()
    c1 = MCPClient("a", _dummy_factory())
    c2 = MCPClient("b", _dummy_factory())
    manager.register(c1)
    manager.register(c2)

    cb = default_logging_callback("shared")
    manager.set_logging_callback(cb, level="error")

    assert c1.logging_callback is cb
    assert c1.logging_level == "error"
    assert c2.logging_callback is cb
    assert c2.logging_level == "error"


# ---------------------------------------------------------------------------
# Session factory integration: callback + setLevel both wired
# ---------------------------------------------------------------------------


def test_session_factory_attaches_logging_callback_and_calls_setlevel(
    monkeypatch,
) -> None:
    pytest.importorskip("mcp")
    from topsport_agent.mcp import client as client_mod

    captured: dict[str, Any] = {}

    class _FakeClientSession:
        def __init__(
            self, _r, _w, *, list_roots_callback=None, logging_callback=None, **_
        ):
            captured["logging_callback"] = logging_callback
            captured["list_roots_callback"] = list_roots_callback

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **kw):
            return None

        async def initialize(self):
            return None

        async def set_logging_level(self, level: str) -> None:
            captured["set_logging_level"] = level

    @contextlib.asynccontextmanager
    async def _fake_stdio_client(_params):
        yield (None, None)

    class _FakeStdioParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def _fake_import(name: str, *_):
        if name == "mcp":
            return type("M", (), {
                "ClientSession": _FakeClientSession,
                "StdioServerParameters": _FakeStdioParams,
            })
        if name == "mcp.client.stdio":
            return type("S", (), {"stdio_client": _fake_stdio_client})
        raise ImportError(name)

    monkeypatch.setattr(
        client_mod, "importlib",
        type("I", (), {"import_module": staticmethod(_fake_import)}),
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)

    seen_params: list[Any] = []

    async def my_cb(params: Any) -> None:
        seen_params.append(params)

    client.set_logging_callback(my_cb, level="warning")

    async def _run() -> None:
        async with client._session_factory():
            pass

    asyncio.run(_run())

    # callback 是 _safe_logging_cb 包装，调用它应转发到原 my_cb
    wrapped = captured["logging_callback"]
    assert wrapped is not None
    asyncio.run(wrapped(_FakeLogParams("info", None, "test")))
    assert len(seen_params) == 1
    assert captured["set_logging_level"] == "warning"


# ---------------------------------------------------------------------------
# ServerConfig env wiring + _build_mcp_manager
# ---------------------------------------------------------------------------


def test_server_config_default_mcp_log_level_empty() -> None:
    cfg = ServerConfig()
    assert cfg.mcp_log_level == ""


def test_server_config_from_env_reads_mcp_log_level(monkeypatch) -> None:
    monkeypatch.setenv("MCP_LOG_LEVEL", "WARNING")
    cfg = ServerConfig.from_env()
    assert cfg.mcp_log_level == "warning"  # 标准化为小写


def test_build_mcp_manager_attaches_logging_callback_per_client() -> None:
    cfg = ServerConfig(
        enable_brave_search=True, brave_api_key="k", mcp_log_level="info",
    )
    manager = _build_mcp_manager(cfg)
    assert manager is not None
    brave = manager.get("brave-search")
    assert brave is not None
    assert brave.logging_callback is not None
    assert brave.logging_level == "info"


@pytest.mark.parametrize("level", ["", "off", "none"])
def test_build_mcp_manager_log_level_disabled(level: str) -> None:
    cfg = ServerConfig(
        enable_brave_search=True, brave_api_key="k", mcp_log_level=level,
    )
    manager = _build_mcp_manager(cfg)
    assert manager is not None
    brave = manager.get("brave-search")
    assert brave is not None
    assert brave.logging_callback is None
    assert brave.logging_level is None


def test_build_mcp_manager_invalid_log_level_fails_fast() -> None:
    cfg = ServerConfig(
        enable_brave_search=True, brave_api_key="k", mcp_log_level="LOUD",
    )
    with pytest.raises(RuntimeError, match="MCP_LOG_LEVEL must be one of"):
        _build_mcp_manager(cfg)


# ---------------------------------------------------------------------------
# Exception isolation + meta dispatch + HTTP transport
# ---------------------------------------------------------------------------


async def test_default_callback_dispatches_meta_into_extra(caplog) -> None:
    cb = default_logging_callback("c")
    with caplog.at_level("INFO", logger="topsport_agent.mcp.server.c"):
        await cb(_FakeLogParams("info", None, "x"))  # no meta
    rec_no_meta = [r for r in caplog.records if "x" == r.message[:1] or "x" in r.message]
    assert rec_no_meta
    assert not hasattr(rec_no_meta[0], "mcp_meta")

    caplog.clear()

    class _ParamsWithMeta:
        level = "info"
        logger = None
        data = "y"
        meta = {"trace_id": "abc"}

    with caplog.at_level("INFO", logger="topsport_agent.mcp.server.c"):
        await cb(_ParamsWithMeta())
    rec_meta = [r for r in caplog.records if hasattr(r, "mcp_meta")]
    assert rec_meta
    assert rec_meta[0].mcp_meta == {"trace_id": "abc"}  # type: ignore[attr-defined]


def test_callback_exception_isolated_from_session(monkeypatch) -> None:
    """Bad user callback must not crash the SDK notification loop:
    session_factory wraps it in a try/except shim that logs WARNING."""
    pytest.importorskip("mcp")
    from topsport_agent.mcp import client as client_mod

    captured: dict[str, Any] = {}

    class _FakeClientSession:
        def __init__(self, _r, _w, *, logging_callback=None, **_):
            captured["wrapped_cb"] = logging_callback

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **kw):
            return None

        async def initialize(self):
            return None

    @contextlib.asynccontextmanager
    async def _fake_stdio_client(_p):
        yield (None, None)

    class _FakeStdio:
        def __init__(self, **kw):
            pass

    def _fake_import(name: str, *_):
        if name == "mcp":
            return type("M", (), {
                "ClientSession": _FakeClientSession,
                "StdioServerParameters": _FakeStdio,
            })
        if name == "mcp.client.stdio":
            return type("S", (), {"stdio_client": _fake_stdio_client})
        raise ImportError(name)

    monkeypatch.setattr(
        client_mod, "importlib",
        type("I", (), {"import_module": staticmethod(_fake_import)}),
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)

    async def _bad_cb(_params):
        raise RuntimeError("user callback exploded")

    client.set_logging_callback(_bad_cb)

    async def _run():
        async with client._session_factory():
            pass

    asyncio.run(_run())
    wrapped = captured["wrapped_cb"]
    assert wrapped is not None
    # 调用 wrapped 不应抛（异常被隔离）
    asyncio.run(wrapped(_FakeLogParams("info", None, "x")))


def test_session_factory_attaches_logging_for_http_transport(monkeypatch) -> None:
    """对称 STDIO 测试：HTTP 分支也必须 attach logging_callback + 调 set_logging_level。"""
    pytest.importorskip("mcp")
    from topsport_agent.mcp import client as client_mod

    captured: dict[str, Any] = {}

    class _FakeClientSession:
        def __init__(self, _r, _w, *, logging_callback=None, **_):
            captured["logging_callback"] = logging_callback

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **kw):
            return None

        async def initialize(self):
            return None

        async def set_logging_level(self, level: str):
            captured["set_logging_level"] = level

    @contextlib.asynccontextmanager
    async def _fake_streamable_http(*, url, http_client):
        yield (None, None)

    class _FakeAsyncClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **kw):
            return None

    def _fake_import(name: str, *_):
        if name == "mcp":
            return type("M", (), {
                "ClientSession": _FakeClientSession,
                "StdioServerParameters": object,
            })
        if name == "mcp.client.streamable_http":
            return type("H", (), {"streamable_http_client": _fake_streamable_http})
        if name == "httpx":
            return type("X", (), {"AsyncClient": _FakeAsyncClient})
        raise ImportError(name)

    monkeypatch.setattr(
        client_mod, "importlib",
        type("I", (), {"import_module": staticmethod(_fake_import)}),
    )

    cfg = MCPServerConfig(
        name="remote",
        transport=MCPTransport.HTTP,
        url="https://example.com/mcp",
    )
    client = MCPClient.from_config(cfg)
    cb = default_logging_callback("remote")
    client.set_logging_callback(cb, level="error")

    async def _run():
        async with client._session_factory():
            pass

    asyncio.run(_run())

    assert captured["logging_callback"] is not None
    assert captured["set_logging_level"] == "error"


def test_post_init_warns_when_sdk_lacks_set_logging_level(monkeypatch, caplog) -> None:
    """SDK 旧版本无 set_logging_level 时记 INFO 日志而非 WARNING（区分 SDK 缺失 vs 服务端拒绝）。"""
    pytest.importorskip("mcp")
    from topsport_agent.mcp import client as client_mod

    class _OldSession:
        def __init__(self, _r, _w, *, logging_callback=None, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **kw):
            return None

        async def initialize(self):
            return None
        # 故意不实现 set_logging_level

    @contextlib.asynccontextmanager
    async def _fake_stdio_client(_p):
        yield (None, None)

    class _FakeStdio:
        def __init__(self, **kw):
            pass

    def _fake_import(name: str, *_):
        if name == "mcp":
            return type("M", (), {
                "ClientSession": _OldSession,
                "StdioServerParameters": _FakeStdio,
            })
        if name == "mcp.client.stdio":
            return type("S", (), {"stdio_client": _fake_stdio_client})
        raise ImportError(name)

    monkeypatch.setattr(
        client_mod, "importlib",
        type("I", (), {"import_module": staticmethod(_fake_import)}),
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    client.set_logging_level("info")

    async def _run():
        async with client._session_factory():
            pass

    with caplog.at_level("INFO", logger="topsport_agent.mcp.client"):
        asyncio.run(_run())

    assert any("SDK lacks set_logging_level" in r.message for r in caplog.records)
