"""MCP `roots` capability tests.

覆盖：
- Root 构造校验 file:// 前缀
- path_to_root 把 Path 转 file:// URI + name 默认 basename
- static_roots provider 每次返回新 list（不让 caller mutate 影响后续）
- call_roots_provider 兼容 sync / async provider，非 list 抛 TypeError
- MCPClient.set_roots_provider 持久存储；_list_roots_callback 路由当前 provider
- MCPClient _list_roots_callback 在无 provider 时返回 ErrorData(method not found)
- MCPClient _list_roots_callback 在 provider 抛异常时返回 ErrorData(internal)
- MCPManager.set_roots_provider 批量应用到所有 client
- ServerConfig.mcp_roots 从 env MCP_ROOTS 解析（冒号分隔，PATH 风格）
- _build_mcp_manager 在 mcp_roots 非空时注册 static_roots provider
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.mcp import (
    MCPClient,
    MCPManager,
    MCPServerConfig,
    Root,
    path_to_root,
    static_roots,
)
from topsport_agent.mcp.roots import call_roots_provider
from topsport_agent.mcp.types import MCPTransport
from topsport_agent.server.app import _build_mcp_manager
from topsport_agent.server.config import ServerConfig, _parse_path_list


# ---------------------------------------------------------------------------
# Root dataclass + helpers
# ---------------------------------------------------------------------------


def test_root_requires_file_uri_prefix() -> None:
    Root(uri="file:///tmp/x")  # OK (三斜杠)
    Root(uri="file://localhost/tmp/x")  # OK (localhost form)
    with pytest.raises(ValueError, match="local file URI"):
        Root(uri="https://example.com/")
    with pytest.raises(ValueError, match="local file URI"):
        Root(uri="/tmp/x")
    with pytest.raises(ValueError, match="local file URI"):
        Root(uri="file://otherhost/x")


def test_root_optional_name_and_meta_defaults() -> None:
    r = Root(uri="file:///x")
    assert r.name is None
    assert r.meta is None


def test_path_to_root_resolves_and_defaults_name(tmp_path: Path) -> None:
    target = tmp_path / "myproj"
    target.mkdir()
    r = path_to_root(target)
    assert r.uri == target.resolve().as_uri()
    assert r.name == "myproj"

    # explicit name overrides default
    r2 = path_to_root(target, name="Project Foo")
    assert r2.name == "Project Foo"


def test_path_to_root_expands_user(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    r = path_to_root("~/")
    assert r.uri.startswith("file://")
    assert tmp_path.resolve().as_uri() == r.uri


def test_static_roots_provider_returns_fresh_list_each_call() -> None:
    seed = [Root(uri="file:///a"), Root(uri="file:///b")]
    provider = static_roots(seed)

    out1 = provider()
    out2 = provider()
    assert out1 == out2
    assert out1 is not out2  # fresh list each call

    # mutating return doesn't affect provider's snapshot
    out1.append(Root(uri="file:///c"))
    out3 = provider()
    assert len(out3) == 2


def test_static_roots_snapshot_does_not_alias_input_after_construction() -> None:
    seed = [Root(uri="file:///a")]
    provider = static_roots(seed)
    seed.append(Root(uri="file:///evil"))  # mutate input post-construction

    assert len(provider()) == 1, "static_roots should snapshot at construction"


# ---------------------------------------------------------------------------
# call_roots_provider sync/async dispatch
# ---------------------------------------------------------------------------


async def test_call_roots_provider_handles_sync_provider() -> None:
    out = await call_roots_provider(lambda: [Root(uri="file:///x")])
    assert out == [Root(uri="file:///x")]


async def test_call_roots_provider_handles_async_provider() -> None:
    async def async_provider() -> list[Root]:
        await asyncio.sleep(0)
        return [Root(uri="file:///async")]

    out = await call_roots_provider(async_provider)
    assert out == [Root(uri="file:///async")]


async def test_call_roots_provider_rejects_non_list() -> None:
    with pytest.raises(TypeError, match="must return list"):
        await call_roots_provider(lambda: "not-a-list")  # type: ignore[arg-type,return-value]


# ---------------------------------------------------------------------------
# MCPClient roots_provider integration
# ---------------------------------------------------------------------------


def _dummy_factory() -> Any:
    @contextlib.asynccontextmanager
    async def factory():
        yield None

    return factory


async def test_client_default_has_no_roots_provider() -> None:
    client = MCPClient("s", _dummy_factory())
    assert client.roots_provider is None


async def test_client_set_roots_provider_replaces_provider() -> None:
    client = MCPClient("s", _dummy_factory())

    p1 = static_roots([Root(uri="file:///a")])
    client.set_roots_provider(p1)
    assert client.roots_provider is p1

    client.set_roots_provider(None)
    assert client.roots_provider is None


async def test_list_roots_callback_returns_error_when_no_provider() -> None:
    """spec: 没声明 roots 能力时 server 不应调，但万一还是调了，返回 -32601。"""
    pytest.importorskip("mcp")

    client = MCPClient("s", _dummy_factory())
    result = await client._list_roots_callback(_context=None)

    from mcp.types import ErrorData

    assert isinstance(result, ErrorData)
    assert result.code == -32601


async def test_list_roots_callback_returns_roots_from_provider() -> None:
    pytest.importorskip("mcp")

    client = MCPClient("s", _dummy_factory())
    client.set_roots_provider(
        static_roots([
            Root(uri="file:///proj", name="Project"),
            Root(uri="file:///docs"),
        ])
    )
    result = await client._list_roots_callback(_context=None)

    from mcp.types import ListRootsResult

    assert isinstance(result, ListRootsResult)
    assert len(result.roots) == 2
    assert str(result.roots[0].uri) == "file:///proj"
    assert result.roots[0].name == "Project"
    assert str(result.roots[1].uri) == "file:///docs"


async def test_list_roots_callback_returns_error_when_provider_raises() -> None:
    pytest.importorskip("mcp")

    client = MCPClient("s", _dummy_factory())

    def boom() -> list[Root]:
        raise RuntimeError("provider exploded")

    client.set_roots_provider(boom)
    result = await client._list_roots_callback(_context=None)

    from mcp.types import ErrorData

    assert isinstance(result, ErrorData)
    assert result.code == -32603
    assert "provider exploded" in result.message


# ---------------------------------------------------------------------------
# MCPManager.set_roots_provider batch
# ---------------------------------------------------------------------------


async def test_manager_set_roots_provider_applies_to_all_clients() -> None:
    manager = MCPManager()
    c1 = MCPClient("a", _dummy_factory())
    c2 = MCPClient("b", _dummy_factory())
    manager.register(c1)
    manager.register(c2)

    provider = static_roots([Root(uri="file:///x")])
    manager.set_roots_provider(provider)

    assert c1.roots_provider is provider
    assert c2.roots_provider is provider

    manager.set_roots_provider(None)
    assert c1.roots_provider is None
    assert c2.roots_provider is None


# ---------------------------------------------------------------------------
# ServerConfig env wiring + _build_mcp_manager integration
# ---------------------------------------------------------------------------


def test_parse_path_list_splits_on_colon() -> None:
    assert _parse_path_list("/a:/b:/c") == ["/a", "/b", "/c"]
    assert _parse_path_list("/single") == ["/single"]
    assert _parse_path_list(None) == []
    assert _parse_path_list("") == []
    assert _parse_path_list("   ") == []
    # 空段被过滤（trailing colon, double colon）
    assert _parse_path_list("/a::/b:") == ["/a", "/b"]


def test_server_config_default_mcp_roots_empty() -> None:
    cfg = ServerConfig()
    assert cfg.mcp_roots == []


def test_server_config_from_env_parses_mcp_roots(tmp_path: Path, monkeypatch) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("MCP_ROOTS", f"{a}:{b}")
    cfg = ServerConfig.from_env()
    assert cfg.mcp_roots == [str(a), str(b)]


def test_build_mcp_manager_registers_roots_provider(tmp_path: Path) -> None:
    """ServerConfig.mcp_roots 非空 + 至少一个 MCP source → manager 上挂 provider。"""
    pytest.importorskip("mcp")

    a = tmp_path / "workspace"
    a.mkdir()
    cfg = ServerConfig(
        enable_brave_search=True,
        brave_api_key="k",
        mcp_roots=[str(a)],
    )
    manager = _build_mcp_manager(cfg)
    assert manager is not None
    brave = manager.get("brave-search")
    assert brave is not None
    assert brave.roots_provider is not None
    roots = brave.roots_provider()
    assert len(roots) == 1
    assert roots[0].uri == a.resolve().as_uri()


def test_build_mcp_manager_no_roots_provider_when_mcp_roots_empty() -> None:
    cfg = ServerConfig(enable_brave_search=True, brave_api_key="k", mcp_roots=[])
    manager = _build_mcp_manager(cfg)
    assert manager is not None
    brave = manager.get("brave-search")
    assert brave is not None
    assert brave.roots_provider is None


# ---------------------------------------------------------------------------
# Integration: factory attaches list_roots_callback to ClientSession only when
# a provider is set — kills the regression risk where set_roots_provider
# silently no-ops because the factory was built with an old client snapshot.
# ---------------------------------------------------------------------------


def test_real_session_factory_attaches_callback_only_when_provider_set(
    monkeypatch,
) -> None:
    pytest.importorskip("mcp")
    from topsport_agent.mcp import client as client_mod

    captured: dict[str, Any] = {}

    class _FakeClientSession:
        def __init__(self, _read, _write, *, list_roots_callback=None, **_):
            captured["list_roots_callback"] = list_roots_callback

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **kw):
            return None

        async def initialize(self):
            return None

    @contextlib.asynccontextmanager
    async def _fake_stdio_client(_params):
        yield (None, None)

    class _FakeStdioParams:
        def __init__(self, **kwargs):
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

    cfg = MCPServerConfig(
        name="x", transport=MCPTransport.STDIO, command="x",
    )
    client = MCPClient.from_config(cfg)

    # Provider 未设：callback 应不 attach（None 透传）
    async def _run_no_provider() -> None:
        async with client._session_factory():
            pass

    asyncio.run(_run_no_provider())
    assert captured["list_roots_callback"] is None

    # 设 provider 后：callback 是 client._list_roots_callback
    client.set_roots_provider(static_roots([Root(uri="file:///x")]))

    async def _run_with_provider() -> None:
        async with client._session_factory():
            pass

    asyncio.run(_run_with_provider())
    # bound method 每次 attribute 访问生成新对象，改用 == 而非 is
    assert captured["list_roots_callback"] == client._list_roots_callback

    # 取消 provider：callback 重新变 None（验证延迟绑定真的 dynamic）
    client.set_roots_provider(None)

    async def _run_after_clear() -> None:
        async with client._session_factory():
            pass

    asyncio.run(_run_after_clear())
    assert captured["list_roots_callback"] is None


def test_static_roots_rejects_non_root_elements() -> None:
    with pytest.raises(TypeError, match="must be a Root"):
        static_roots(["file:///x"])  # type: ignore[list-item]


def test_root_rejects_remote_host_uri() -> None:
    """spec local-file form: file:/// 或 file://localhost/. 远端 host 必须 raise。"""
    with pytest.raises(ValueError, match="local file URI"):
        Root(uri="file://otherhost/path")
