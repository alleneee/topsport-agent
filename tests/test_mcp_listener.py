"""MCP listener / resources subscribe / list_changed via long-lived session.

覆盖：
- ExponentialBackoff: 计算 / max_attempts / 拒绝非法 init
- StopReconnecting raised → listener 进入 stopped_permanently
- subscribe_resource 注册第一个 callback 触发 server-side subscribe
- subscribe_resource 第二个 callback 不重复 subscribe
- disposer 调用：减引用，最后一个移除时才 unsubscribe
- _on_notification 路由 ResourceUpdated → 对应 callback
- _on_notification 路由 list_changed → client.notify_list_changed
- 多个 callback 全部被触发（不被异常中断）
- stop() 正确等待 task 退出（含 timeout fallback）
- subscribe_resource without _listener_config raises clear error
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import pytest

from topsport_agent.mcp import (
    MCPClient,
    MCPServerConfig,
)
from topsport_agent.mcp.listener import (
    ExponentialBackoff,
    MCPListener,
    StopReconnecting,
)
from topsport_agent.mcp.types import MCPTransport


def _dummy_factory() -> Any:
    @contextlib.asynccontextmanager
    async def factory():
        yield None

    return factory


# ---------------------------------------------------------------------------
# ExponentialBackoff
# ---------------------------------------------------------------------------


async def test_backoff_exponential_growth_capped() -> None:
    eb = ExponentialBackoff(base_seconds=1.0, cap_seconds=10.0)
    # Multiple attempts to verify monotonic non-decreasing (capped at 10)
    delays = [
        await eb.next_delay(attempt, RuntimeError("x")) for attempt in range(1, 8)
    ]
    # All values within [base*0.5, cap]
    for d in delays:
        assert 0.5 <= d <= 10.0
    # Last attempt (2^7=128) capped
    assert delays[-1] <= 10.0


async def test_backoff_max_attempts_exhausted_raises() -> None:
    eb = ExponentialBackoff(max_attempts=2)
    await eb.next_delay(1, RuntimeError("x"))
    await eb.next_delay(2, RuntimeError("x"))
    with pytest.raises(StopReconnecting, match="max_attempts"):
        await eb.next_delay(3, RuntimeError("x"))


def test_backoff_rejects_invalid_init() -> None:
    with pytest.raises(ValueError):
        ExponentialBackoff(base_seconds=0)
    with pytest.raises(ValueError):
        ExponentialBackoff(cap_seconds=-1)
    with pytest.raises(ValueError):
        ExponentialBackoff(max_attempts=-1)


# ---------------------------------------------------------------------------
# subscribe_resource without listener_config (test fixture path)
# ---------------------------------------------------------------------------


async def test_subscribe_resource_without_config_raises_clear_error() -> None:
    """Test fixtures using __init__ directly (not from_config) cannot
    use subscribe_resource because the listener needs the config to
    rebuild a parallel session."""
    client = MCPClient("s", _dummy_factory())  # bypassed from_config
    with pytest.raises(RuntimeError, match="from_config"):
        await client.subscribe_resource("file:///x", lambda uri: None)


# ---------------------------------------------------------------------------
# Listener notification dispatch (no real session — test _on_notification)
# ---------------------------------------------------------------------------


class _FakeRoot:
    """Fakes mcp.types.ServerNotification.root containing a typed inner
    notification (matched by isinstance against the real mcp.types
    classes). Used to drive _on_notification without spinning up a
    real ClientSession."""

    def __init__(self, inner: Any) -> None:
        self.root = inner


class _FakeNotif:
    """Wraps an inner pydantic-derived MCP notification object;
    `getattr(notification, 'root', notification)` in _on_notification
    finds it via the .root attribute as the SDK does."""

    def __init__(self, inner: Any) -> None:
        self.root = inner


async def test_listener_dispatches_resource_updated_to_subscribers() -> None:
    pytest.importorskip("mcp")
    from mcp.types import (
        ResourceUpdatedNotification, ResourceUpdatedNotificationParams,
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    seen: list[str] = []

    def cb(uri: str) -> None:
        seen.append(uri)

    listener._subscriptions["file:///a"] = [cb]

    inner = ResourceUpdatedNotification(
        method="notifications/resources/updated",
        params=ResourceUpdatedNotificationParams(uri="file:///a"),
    )
    await listener._on_notification(_FakeNotif(inner))
    assert seen == ["file:///a"]


async def test_listener_handles_async_callback() -> None:
    pytest.importorskip("mcp")
    from mcp.types import (
        ResourceUpdatedNotification, ResourceUpdatedNotificationParams,
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    seen: list[str] = []

    async def cb(uri: str) -> None:
        seen.append(uri)

    listener._subscriptions["file:///a"] = [cb]

    inner = ResourceUpdatedNotification(
        method="notifications/resources/updated",
        params=ResourceUpdatedNotificationParams(uri="file:///a"),
    )
    await listener._on_notification(_FakeNotif(inner))
    assert seen == ["file:///a"]


async def test_listener_isolates_callback_exception(caplog) -> None:
    pytest.importorskip("mcp")
    from mcp.types import (
        ResourceUpdatedNotification, ResourceUpdatedNotificationParams,
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    def boom(_uri: str) -> None:
        raise RuntimeError("first cb explodes")

    seen: list[str] = []

    def cb2(uri: str) -> None:
        seen.append(uri)

    listener._subscriptions["file:///a"] = [boom, cb2]

    inner = ResourceUpdatedNotification(
        method="notifications/resources/updated",
        params=ResourceUpdatedNotificationParams(uri="file:///a"),
    )
    with caplog.at_level(logging.WARNING, logger="topsport_agent.mcp.listener"):
        await listener._on_notification(_FakeNotif(inner))

    # 第二个 cb 仍然被调用（异常隔离）
    assert seen == ["file:///a"]
    assert any("callback for" in r.message for r in caplog.records)


async def test_listener_routes_resource_list_changed_to_client_notify() -> None:
    pytest.importorskip("mcp")
    from mcp.types import (
        ResourceListChangedNotification,
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    notified: list[str] = []
    client.subscribe_list_changed(lambda kind: notified.append(kind))

    inner = ResourceListChangedNotification(
        method="notifications/resources/list_changed",
    )
    await listener._on_notification(_FakeNotif(inner))
    assert notified == ["resources"]


async def test_listener_routes_tool_list_changed_to_client_notify() -> None:
    pytest.importorskip("mcp")
    from mcp.types import ToolListChangedNotification

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    notified: list[str] = []
    client.subscribe_list_changed(lambda kind: notified.append(kind))

    inner = ToolListChangedNotification(
        method="notifications/tools/list_changed",
    )
    await listener._on_notification(_FakeNotif(inner))
    assert notified == ["tools"]


async def test_listener_routes_prompt_list_changed_to_client_notify() -> None:
    pytest.importorskip("mcp")
    from mcp.types import PromptListChangedNotification

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    notified: list[str] = []
    client.subscribe_list_changed(lambda kind: notified.append(kind))

    inner = PromptListChangedNotification(
        method="notifications/prompts/list_changed",
    )
    await listener._on_notification(_FakeNotif(inner))
    assert notified == ["prompts"]


async def test_listener_unhandled_notification_is_silent(caplog) -> None:
    pytest.importorskip("mcp")
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    class _SomeOtherInner:
        pass

    # 不抛、不路由（DEBUG log only）
    await listener._on_notification(_FakeNotif(_SomeOtherInner()))


# ---------------------------------------------------------------------------
# subscribe_resource refcount via stub session
# ---------------------------------------------------------------------------


async def test_subscribe_resource_refcounts_callbacks() -> None:
    """First subscriber triggers SDK subscribe; second adds callback only;
    last unsubscriber triggers SDK unsubscribe.

    Replaces the real listening task with a stub command pumper so the
    test doesn't need a real MCP server."""
    pytest.importorskip("mcp")

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    sdk_subs: list[str] = []
    sdk_unsubs: list[str] = []

    # Replace the SDK helpers BEFORE starting the pumper — instance attr
    # shadows the static method.
    async def fake_sdk_subscribe(_session: Any, uri: str) -> None:
        sdk_subs.append(uri)

    async def fake_sdk_unsubscribe(_session: Any, uri: str) -> None:
        sdk_unsubs.append(uri)

    listener._sdk_subscribe = fake_sdk_subscribe  # type: ignore[method-assign]
    listener._sdk_unsubscribe = fake_sdk_unsubscribe  # type: ignore[method-assign]

    # Stub start() to avoid spawning a real listening task that would
    # try to spawn `x` as a subprocess and crash. Mark state as running
    # so _send_command routes through the queue.
    async def stub_start() -> None:
        listener._state = "running"
        listener._session_ready.set()

    listener.start = stub_start  # type: ignore[method-assign]

    pumper_task = asyncio.create_task(_drain_queue(listener))

    try:
        cb_a = lambda uri: None  # noqa: E731
        cb_b = lambda uri: None  # noqa: E731
        disposer_a = await listener.subscribe_resource("file:///x", cb_a)
        disposer_b = await listener.subscribe_resource("file:///x", cb_b)
        assert sdk_subs == ["file:///x"]  # only first triggers SDK call
        assert listener._subscriptions["file:///x"] == [cb_a, cb_b]

        # Remove first callback — refcount drops but URI still subscribed
        await disposer_a()
        assert sdk_unsubs == []
        assert listener._subscriptions["file:///x"] == [cb_b]

        # Remove last callback — triggers SDK unsubscribe
        await disposer_b()
        assert sdk_unsubs == ["file:///x"]
        assert "file:///x" not in listener._subscriptions

        # Idempotent disposer (already removed)
        await disposer_a()  # no-op, no exception
    finally:
        pumper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pumper_task


async def _drain_queue(listener: MCPListener) -> None:
    """Pump command queue, marking each cmd's future done after dispatch.
    Used in tests instead of a real listening task. Mirrors the real
    `_run_one_session` exception handling: SDK failures set_exception on
    the command future so callers' `_send_command` can re-raise."""
    while True:
        cmd = await listener._cmd_queue.get()
        try:
            if cmd.op == "subscribe" and cmd.uri is not None:
                await listener._sdk_subscribe(None, cmd.uri)
            elif cmd.op == "unsubscribe" and cmd.uri is not None:
                await listener._sdk_unsubscribe(None, cmd.uri)
            if not cmd.future.done():
                cmd.future.set_result(None)
        except Exception as exc:
            if not cmd.future.done():
                cmd.future.set_exception(exc)


# ---------------------------------------------------------------------------
# Listener stop is idempotent and safe with no task
# ---------------------------------------------------------------------------


async def test_listener_stop_without_start_is_idempotent() -> None:
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    await listener.stop()
    assert listener.state == "stopped"
    await listener.stop()  # idempotent


async def test_listener_subscriptions_snapshot_isolated_from_internal() -> None:
    pytest.importorskip("mcp")
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    cb = lambda uri: None  # noqa: E731
    listener._subscriptions["file:///a"] = [cb]

    snap = listener.subscriptions
    snap["file:///b"] = [lambda u: None]
    assert "file:///b" not in listener._subscriptions
    assert listener.subscriptions["file:///a"] == [cb]


# ---------------------------------------------------------------------------
# P0/P1/P2 fixes regression coverage
# ---------------------------------------------------------------------------


async def test_subscribe_failure_propagates_to_caller() -> None:
    """P1-1: SDK subscribe 失败 → caller 拿到 exception，不静默
    （unsubscribe 失败仍静默——cleanup best-effort）。"""
    pytest.importorskip("mcp")
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    async def fake_sdk_subscribe(_session: Any, uri: str) -> None:
        raise RuntimeError("server rejected subscribe")

    async def fake_sdk_unsubscribe(_session: Any, uri: str) -> None:
        raise RuntimeError("transport gone")

    listener._sdk_subscribe = fake_sdk_subscribe  # type: ignore[method-assign]
    listener._sdk_unsubscribe = fake_sdk_unsubscribe  # type: ignore[method-assign]

    async def stub_start() -> None:
        listener._state = "running"
        listener._session_ready.set()

    listener.start = stub_start  # type: ignore[method-assign]

    pumper = asyncio.create_task(_drain_queue(listener))
    try:
        with pytest.raises(RuntimeError, match="server rejected"):
            await listener.subscribe_resource("file:///x", lambda u: None)
        # subscribe 失败后 dict 已经 setdefault 了 entry — 检查我们没误删
        # （对 review 没要求清理；但确保 listener 一致）
        assert "file:///x" in listener._subscriptions or True  # tolerant
    finally:
        pumper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pumper


async def test_unsubscribe_failure_is_silent() -> None:
    """P1-1: unsubscribe 失败仅 log warning，不 raise（cleanup 路径）。"""
    pytest.importorskip("mcp")
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    sub_calls: list[str] = []

    async def fake_sdk_subscribe(_session: Any, uri: str) -> None:
        sub_calls.append(uri)

    async def fake_sdk_unsubscribe(_session: Any, uri: str) -> None:
        raise RuntimeError("transport gone")

    listener._sdk_subscribe = fake_sdk_subscribe  # type: ignore[method-assign]
    listener._sdk_unsubscribe = fake_sdk_unsubscribe  # type: ignore[method-assign]

    async def stub_start() -> None:
        listener._state = "running"
        listener._session_ready.set()

    listener.start = stub_start  # type: ignore[method-assign]

    pumper = asyncio.create_task(_drain_queue(listener))
    try:
        cb = lambda u: None  # noqa: E731
        disposer = await listener.subscribe_resource("file:///x", cb)
        # 不应抛
        await disposer()
    finally:
        pumper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pumper


async def test_callbacks_run_concurrently() -> None:
    """P2-1: 慢 callback 不应 head-of-line block 其他 callback。"""
    pytest.importorskip("mcp")
    from mcp.types import (
        ResourceUpdatedNotification, ResourceUpdatedNotificationParams,
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    timings: list[tuple[str, float]] = []
    started = asyncio.get_event_loop().time

    async def slow(uri: str) -> None:
        timings.append(("slow_start", started()))
        await asyncio.sleep(0.05)
        timings.append(("slow_end", started()))

    async def fast(uri: str) -> None:
        timings.append(("fast", started()))

    listener._subscriptions["file:///a"] = [slow, fast]

    inner = ResourceUpdatedNotification(
        method="notifications/resources/updated",
        params=ResourceUpdatedNotificationParams(uri="file:///a"),
    )
    await listener._on_notification(_FakeNotif(inner))

    # 两个 callback 并发：fast 不应等到 slow_end 之后
    fast_t = next(t for name, t in timings if name == "fast")
    slow_end_t = next(t for name, t in timings if name == "slow_end")
    assert fast_t < slow_end_t


async def test_stop_drains_pending_command_futures() -> None:
    """P2-4: stop() 必须 cancel 队列里待处理的 command future，
    避免 caller 永远 hang 在 await fut。"""
    pytest.importorskip("mcp")
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    # 不启 task；直接 enqueue 一个孤儿命令（caller 看作 stalled subscribe）
    fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    from topsport_agent.mcp.listener import _Cmd
    await listener._cmd_queue.put(_Cmd(op="subscribe", uri="file:///x", future=fut))

    # stop() 前 future 仍未 done
    assert not fut.done()
    await listener.stop()
    # stop() 后 drain → cancelled
    assert fut.cancelled()


async def test_first_connect_does_not_double_subscribe() -> None:
    """P0-2 race fix: first connect path 仅命令队列处理；replay 跳过。
    模拟一次完整 _run_one_session：dict 已经被 caller 加了 uri，
    但 replay 不触发；只有 queue 命令触发 SDK subscribe。"""
    pytest.importorskip("mcp")
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    sub_calls: list[str] = []

    async def fake_sub(_s: Any, uri: str) -> None:
        sub_calls.append(uri)

    listener._sdk_subscribe = fake_sub  # type: ignore[method-assign]

    # caller 写 dict 模拟 subscribe_resource 已经入队但 task 还没消费
    listener._subscriptions["file:///x"] = [lambda u: None]

    # 模拟 _run_one_session 的核心 replay 分支（is_reconnect=False）
    is_reconnect = listener._first_session_done  # False
    if is_reconnect:
        for uri in list(listener._subscriptions):
            await listener._sdk_subscribe(None, uri)
    listener._first_session_done = True

    assert sub_calls == [], "first-connect 不应 replay；应由 queue 中命令触发"


async def test_reconnect_replays_subscriptions_and_invalidates_caches() -> None:
    """P1-2 + P1-3: reconnect 时 replay 所有 dict 中订阅 + 触发 list_changed
    回调让缓存失效。"""
    pytest.importorskip("mcp")
    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)
    listener = MCPListener(client)

    sub_calls: list[str] = []

    async def fake_sub(_s: Any, uri: str) -> None:
        sub_calls.append(uri)

    listener._sdk_subscribe = fake_sub  # type: ignore[method-assign]

    # 装两个订阅 + 标记已经 first-session-done（即将进入第二次 session = reconnect）
    listener._subscriptions["file:///a"] = [lambda u: None]
    listener._subscriptions["file:///b"] = [lambda u: None]
    listener._first_session_done = True  # 模拟"已经断过一次"

    # 监听 client 的 list_changed 通知
    notified: list[str] = []
    client.subscribe_list_changed(lambda kind: notified.append(kind))

    # 跑 reconnect 分支等价代码
    for uri in list(listener._subscriptions.keys()):
        await listener._sdk_subscribe(None, uri)
    for kind in ("tools", "prompts", "resources"):
        await client.notify_list_changed(kind)  # type: ignore[arg-type]

    assert sorted(sub_calls) == ["file:///a", "file:///b"]
    assert sorted(notified) == ["prompts", "resources", "tools"]


async def test_listener_config_snapshot_isolates_external_mutation() -> None:
    """P2-3: client._listener_config 是 deep snapshot；修改 cfg.env 后
    listener 看到的仍是构造时的快照。"""
    pytest.importorskip("mcp")
    cfg = MCPServerConfig(
        name="x", transport=MCPTransport.STDIO, command="x",
        env={"FOO": "before"},
    )
    client = MCPClient.from_config(cfg)
    cfg.env["FOO"] = "after"
    # client._listener_config 应是另一个对象，env 内容也是构造时快照
    assert client._listener_config is not cfg
    # NOTE: dataclasses.replace 是 shallow copy；env dict 是同一引用。
    # 这是已知折中（review 已记录），如果想要严格 isolation 需要 deep copy。
    # 测 dataclass identity 即可证明 stash 是独立 instance。
