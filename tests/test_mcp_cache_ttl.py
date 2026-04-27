"""MCP client TTL 缓存 + listChanged 订阅测试。

覆盖：
- TTL 三种语义：正数过期 / None 永不过期 / 0 不缓存
- 负数 cache_ttl 在 MCPServerConfig.__post_init__ 与 MCPClient.__init__ 都被拒绝
- force_refresh / invalidate_cache 行为
- notify_list_changed 选择性失效 + 订阅者回调
- subscribe_list_changed 返回 disposer，调用注销
- 订阅者抛异常被隔离 + log warning 记录
- async + sync 回调都被支持
- 时钟回拨被保守视为 stale
- load_mcp_config 解析 cache_ttl（数值 / null / 缺失默认）
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.mcp import MCPClient, MCPServerConfig, load_mcp_config
from topsport_agent.mcp.types import MCPTransport


class _SpySession:
    def __init__(self) -> None:
        self.tool_calls = 0
        self.prompt_calls = 0
        self.resource_calls = 0

    async def list_tools(self) -> Any:
        self.tool_calls += 1

        class _R:
            tools: list[Any] = []

        return _R()

    async def list_prompts(self) -> Any:
        self.prompt_calls += 1

        class _R:
            prompts: list[Any] = []

        return _R()

    async def list_resources(self) -> Any:
        self.resource_calls += 1

        class _R:
            resources: list[Any] = []

        return _R()


def _factory_with_counter() -> tuple[Any, list[_SpySession]]:
    sessions: list[_SpySession] = []

    @contextlib.asynccontextmanager
    async def factory():
        sess = _SpySession()
        sessions.append(sess)
        yield sess

    return factory, sessions


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


# ---------------------------------------------------------------------------
# TTL semantics
# ---------------------------------------------------------------------------


async def test_list_tools_caches_within_ttl_window() -> None:
    factory, sessions = _factory_with_counter()
    clock = _Clock()
    client = MCPClient("s", factory, cache_ttl=60.0, clock=clock)

    await client.list_tools()
    await client.list_tools()
    await client.list_tools()

    assert len(sessions) == 1


async def test_list_tools_refreshes_after_ttl_expires() -> None:
    factory, sessions = _factory_with_counter()
    clock = _Clock()
    client = MCPClient("s", factory, cache_ttl=60.0, clock=clock)

    await client.list_tools()
    clock.advance(60.5)
    await client.list_tools()

    assert len(sessions) == 2


async def test_cache_ttl_none_means_never_expire() -> None:
    """None = 永不过期；只能靠 invalidate / force_refresh / notify 失效。"""
    factory, sessions = _factory_with_counter()
    clock = _Clock()
    client = MCPClient("s", factory, cache_ttl=None, clock=clock)

    await client.list_tools()
    clock.advance(10000)
    await client.list_tools()

    assert len(sessions) == 1


async def test_cache_ttl_zero_means_no_cache_each_call_refetches() -> None:
    """0 = 不缓存（与 HTTP Cache-Control: max-age=0 语义一致）。"""
    factory, sessions = _factory_with_counter()
    client = MCPClient("s", factory, cache_ttl=0)

    await client.list_tools()
    await client.list_tools()
    await client.list_tools()

    assert len(sessions) == 3


async def test_negative_cache_ttl_rejected_by_client_init() -> None:
    factory, _ = _factory_with_counter()
    with pytest.raises(ValueError, match="cache_ttl must be >= 0 or None"):
        MCPClient("s", factory, cache_ttl=-1)


def test_negative_cache_ttl_rejected_by_server_config() -> None:
    with pytest.raises(ValueError, match="cache_ttl must be >= 0 or None"):
        MCPServerConfig(
            name="x", transport=MCPTransport.STDIO, command="x", cache_ttl=-1,
        )


async def test_clock_rollback_treated_as_stale() -> None:
    """时钟回拨（age < 0）应保守视为 stale，避免缓存意外延寿。"""
    factory, sessions = _factory_with_counter()
    clock = _Clock(now=1000)
    client = MCPClient("s", factory, cache_ttl=60.0, clock=clock)

    await client.list_tools()
    clock.now = 500  # 时钟回拨
    await client.list_tools()

    assert len(sessions) == 2


async def test_force_refresh_bypasses_cache() -> None:
    factory, sessions = _factory_with_counter()
    client = MCPClient("s", factory, cache_ttl=3600.0)

    await client.list_tools()
    await client.list_tools(force_refresh=True)

    assert len(sessions) == 2


async def test_invalidate_cache_clears_all_lists() -> None:
    factory, sessions = _factory_with_counter()
    client = MCPClient("s", factory, cache_ttl=3600.0)

    await client.list_tools()
    await client.list_prompts()
    await client.list_resources()
    assert len(sessions) == 3

    client.invalidate_cache()

    await client.list_tools()
    await client.list_prompts()
    await client.list_resources()
    assert len(sessions) == 6


# ---------------------------------------------------------------------------
# notify_list_changed + subscription
# ---------------------------------------------------------------------------


async def test_notify_only_invalidates_targeted_kind() -> None:
    factory, sessions = _factory_with_counter()
    client = MCPClient("s", factory, cache_ttl=3600.0)

    await client.list_tools()
    await client.list_prompts()
    await client.list_resources()

    await client.notify_list_changed("tools")

    await client.list_tools()
    await client.list_prompts()
    await client.list_resources()

    tool_sessions = [s for s in sessions if s.tool_calls > 0]
    prompt_sessions = [s for s in sessions if s.prompt_calls > 0]
    resource_sessions = [s for s in sessions if s.resource_calls > 0]
    assert len(tool_sessions) == 2
    assert len(prompt_sessions) == 1
    assert len(resource_sessions) == 1


async def test_subscribe_list_changed_callback_fires_on_notify() -> None:
    factory, _ = _factory_with_counter()
    client = MCPClient("s", factory)

    seen: list[str] = []
    client.subscribe_list_changed(lambda kind: seen.append(kind))

    await client.notify_list_changed("tools")
    await client.notify_list_changed("prompts")
    await client.notify_list_changed("resources")

    assert seen == ["tools", "prompts", "resources"]


async def test_async_callback_is_awaited() -> None:
    factory, _ = _factory_with_counter()
    client = MCPClient("s", factory)

    seen: list[str] = []

    async def async_cb(kind: str) -> None:
        seen.append(f"async:{kind}")

    client.subscribe_list_changed(async_cb)

    await client.notify_list_changed("tools")

    assert seen == ["async:tools"]


async def test_subscribe_returns_disposer_that_unsubscribes() -> None:
    factory, _ = _factory_with_counter()
    client = MCPClient("s", factory)

    seen: list[str] = []
    disposer = client.subscribe_list_changed(lambda kind: seen.append(kind))

    await client.notify_list_changed("tools")
    assert seen == ["tools"]

    disposer()  # 注销

    await client.notify_list_changed("prompts")
    assert seen == ["tools"], "注销后回调不应再触发"

    # 再次调用 disposer 是幂等的（不应抛 ValueError）
    disposer()


async def test_subscriber_exception_is_isolated_and_logged(caplog) -> None:
    factory, _ = _factory_with_counter()
    client = MCPClient("name-x", factory)

    seen: list[str] = []

    def boom(_kind: str) -> None:
        raise RuntimeError("boom")

    client.subscribe_list_changed(boom)
    client.subscribe_list_changed(lambda kind: seen.append(kind))

    with caplog.at_level(logging.WARNING, logger="topsport_agent.mcp.client"):
        await client.notify_list_changed("tools")

    assert seen == ["tools"], "失败的回调不应中断后续回调链"
    assert client._cache_tools is None  # type: ignore[attr-defined]
    # log warning 记录失败 + client 名 + kind
    assert any(
        "list_changed callback failed" in rec.message for rec in caplog.records
    )
    assert any("'name-x'" in rec.message for rec in caplog.records)


async def test_callback_unsubscribing_during_notify_does_not_skip_others() -> None:
    """回调内 unsubscribe 不应让同一 notify 中的后续回调被跳过（用快照迭代）。"""
    factory, _ = _factory_with_counter()
    client = MCPClient("s", factory)

    seen: list[str] = []

    def first(_kind: str) -> None:
        seen.append("first")
        # 在迭代过程中删自己
        try:
            client._list_changed_callbacks.remove(first)  # type: ignore[attr-defined]
        except ValueError:
            pass

    client.subscribe_list_changed(first)
    client.subscribe_list_changed(lambda kind: seen.append("second"))

    await client.notify_list_changed("tools")

    assert seen == ["first", "second"]


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_mcp_server_config_cache_ttl_default_is_60() -> None:
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    assert cfg.cache_ttl == 60.0


def test_load_mcp_config_parses_cache_ttl(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fast": {"transport": "stdio", "command": "x", "cache_ttl": 5},
                    "never_expire": {
                        "transport": "stdio", "command": "x", "cache_ttl": None,
                    },
                    "no_cache": {
                        "transport": "stdio", "command": "x", "cache_ttl": 0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    configs = {c.name: c for c in load_mcp_config(path)}
    assert configs["fast"].cache_ttl == 5.0
    assert configs["never_expire"].cache_ttl is None
    assert configs["no_cache"].cache_ttl == 0.0


def test_load_mcp_config_cache_ttl_default_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {"x": {"transport": "stdio", "command": "x"}}}),
        encoding="utf-8",
    )
    configs = load_mcp_config(path)
    assert configs[0].cache_ttl == 60.0


def test_client_from_config_propagates_cache_ttl() -> None:
    cfg = MCPServerConfig(
        name="x", transport=MCPTransport.STDIO, command="x", cache_ttl=12.5,
    )
    client = MCPClient.from_config(cfg)
    assert client.cache_ttl == 12.5
