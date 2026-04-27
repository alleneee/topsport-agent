"""Long-lived MCP listening session for resource subscriptions and
list-change notifications.

Per the MCP spec (2025-11-25):
    - `resources/subscribe(uri)`: client tells server "notify me when
      the resource at this URI changes". Optional pair `unsubscribe(uri)`.
    - `notifications/resources/updated(uri)`: server pushes when a
      subscribed resource changes; **does not contain the new
      content** (hint-then-pull design — client must re-read).
    - `notifications/{tools,prompts,resources}/list_changed`: server
      advertises that its catalog grew/shrank/changed.

The agent's existing "session per call" model can't receive any of
these — sessions exit before notifications arrive. `MCPListener`
solves this by running a dedicated long-lived `ClientSession` in its
own asyncio task, with two channels into it:
    - command queue (subscribe/unsubscribe RPCs from external tasks
      → enqueued and awaited via futures, so the listening task
      remains the sole owner of the session's cancel scope)
    - message_handler callback (server pushes notifications → routed
      to per-uri callbacks for `updated` events, or to the client's
      `notify_list_changed` for list_change events)

Reconnect strategy: a `ReconnectStrategy` Protocol pluggable; default
is exponential backoff capped at 60s. Operators with stricter
requirements (immediate retry / fail-fast / circuit breaker) plug
their own.

Architectural note: this listener does NOT replace the per-call
short-lived sessions used by `list_tools` / `call_tool` — those still
need their own task-local cancel scope to allow concurrent ReAct
steps. The listener is a parallel session dedicated to receiving
notifications. Most MCP servers tolerate multi-session clients (each
session is a separate connection); stdio transport will spawn two
subprocesses, which is the documented cost.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .client import MCPClient
    from .roots import RootsProvider

_logger = logging.getLogger(__name__)

# updated callback gets the bare URI; spec dictates content is fetched
# separately. Sync OR async; listener dispatches via inspect.isawaitable.
ResourceUpdatedCallback = Callable[[str], "Awaitable[None] | None"]


@runtime_checkable
class ReconnectStrategy(Protocol):
    """Decides what to do after the listening session dies (server
    crash, network drop, transport error). Returns the seconds to wait
    before reconnecting, or raises `StopReconnecting` to give up.
    """

    async def next_delay(self, attempt: int, last_exception: BaseException) -> float:
        """`attempt` starts at 1 for the first reconnect attempt."""
        ...


class StopReconnecting(Exception):
    """Raised by ReconnectStrategy.next_delay to permanently abort the
    listener (it transitions to STOPPED)."""


class ExponentialBackoff:
    """Default reconnect: capped exponential backoff with full jitter.

    delay = min(base * 2^(attempt-1), cap), then `* random.uniform(0.5, 1)`
    so multiple clients reconnecting after a shared outage don't
    thunder-herd the server.

    `max_attempts` defaults to 20 (≈ 10–20 minutes of retries at
    cap_seconds=60). This is fail-soft default: a permanently dead server
    eventually causes the listener to give up rather than spamming
    log + spawning subprocesses indefinitely (review P0-1: infinite
    retry is a self-DoS waiting to happen). Operators who actually want
    indefinite retry — e.g. high-availability deployments where the
    server fleet is expected to come back — pass `max_attempts=None`
    explicitly.
    """

    def __init__(
        self,
        *,
        base_seconds: float = 1.0,
        cap_seconds: float = 60.0,
        max_attempts: int | None = 20,
    ) -> None:
        if base_seconds <= 0 or cap_seconds <= 0:
            raise ValueError("base/cap must be > 0")
        if max_attempts is not None and max_attempts < 0:
            raise ValueError("max_attempts must be >= 0 or None")
        self._base = base_seconds
        self._cap = cap_seconds
        self._max_attempts = max_attempts

    async def next_delay(
        self, attempt: int, last_exception: BaseException,
    ) -> float:
        del last_exception
        if self._max_attempts is not None and attempt > self._max_attempts:
            raise StopReconnecting(
                f"max_attempts={self._max_attempts} exhausted"
            )
        capped = min(self._base * (2 ** (attempt - 1)), self._cap)
        return capped * random.uniform(0.5, 1.0)


# ---------------------------------------------------------------------------
# Listener state machine
# ---------------------------------------------------------------------------


ListenerState = Literal["stopped", "starting", "running", "reconnecting", "stopped_permanently"]


@dataclass(slots=True)
class _Cmd:
    """Internal: command from external task → listening task."""

    op: Literal["subscribe", "unsubscribe", "shutdown"]
    uri: str | None
    future: asyncio.Future[None]


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------


class MCPListener:
    """Manages a long-lived listening MCP session for one MCPClient.

    Lifecycle:
        - lazy: `start()` is idempotent and called automatically by
          `subscribe_resource` first time it runs
        - running: own background task drives one `ClientSession`,
          reading commands off a queue and dispatching server
          notifications to callbacks
        - shutdown: `stop()` enqueues a shutdown command, awaits the
          task to exit cleanly within `shutdown_timeout`

    Subscriptions are tracked on the listener; if the session dies and
    reconnects, the listener replays its subscription set on the new
    session — operators don't lose subscriptions across server
    restarts.
    """

    def __init__(
        self,
        client: "MCPClient",
        *,
        reconnect: ReconnectStrategy | None = None,
        shutdown_timeout: float = 5.0,
    ) -> None:
        self._client = client
        self._reconnect = reconnect or ExponentialBackoff()
        self._shutdown_timeout = shutdown_timeout
        self._task: asyncio.Task[None] | None = None
        self._cmd_queue: asyncio.Queue[_Cmd] = asyncio.Queue()
        self._subscriptions: dict[str, list[ResourceUpdatedCallback]] = {}
        self._state: ListenerState = "stopped"
        # Set when the inner ClientSession is established; cleared on
        # disconnect. Subscribe commands wait for it to be set before
        # forwarding to the session.
        self._session_ready = asyncio.Event()
        self._stop_reason: BaseException | None = None
        # Race fix (review P0-2): distinguish first connect from reconnect.
        # On first connect, the dict is populated by `subscribe_resource`
        # *and* the same command is enqueued — replay would double-call
        # SDK subscribe. Only replay on reconnect; first-connect path
        # relies on the command queue.
        self._first_session_done = False

    @property
    def state(self) -> ListenerState:
        return self._state

    @property
    def subscriptions(self) -> dict[str, list[ResourceUpdatedCallback]]:
        """Return a snapshot of {uri: [callbacks]}. Caller-mutable, no
        leak to listener internal state."""
        return {uri: list(cbs) for uri, cbs in self._subscriptions.items()}

    # -----------------------------------------------------------------
    # Public API: start / stop / subscribe
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Idempotent. Starts the listening task if not already running."""
        if self._task is not None and not self._task.done():
            return
        self._state = "starting"
        self._task = asyncio.create_task(
            self._run(), name=f"mcp-listener-{self._client.name}",
        )

    async def stop(self) -> None:
        """Send shutdown command and wait for task to exit. Cancels
        outstanding subscribe commands gracefully (their futures get
        a CancelledError)."""
        if self._task is None or self._task.done():
            self._state = "stopped"
            self._drain_pending_commands()
            return
        fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        await self._cmd_queue.put(_Cmd(op="shutdown", uri=None, future=fut))
        try:
            await asyncio.wait_for(self._task, timeout=self._shutdown_timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._state = "stopped"
        # P2-4: outstanding command futures (caller awaiting in _send_command)
        # would hang forever once the listening task is gone. Cancel them.
        self._drain_pending_commands()

    def _drain_pending_commands(self) -> None:
        """Cancel every command in the queue: callers waiting on these
        futures will see CancelledError instead of hanging."""
        while not self._cmd_queue.empty():
            try:
                cmd = self._cmd_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not cmd.future.done():
                cmd.future.cancel()

    async def subscribe_resource(
        self, uri: str, callback: ResourceUpdatedCallback,
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe to `notifications/resources/updated` for `uri`.

        Returns an async disposer; awaiting it unsubscribes (decrements
        ref count; only sends `resources/unsubscribe` to the server when
        the last callback for that URI is removed).

        First call lazily starts the listener.
        """
        await self.start()

        callbacks = self._subscriptions.setdefault(uri, [])
        first = not callbacks
        callbacks.append(callback)

        if first:
            await self._send_command("subscribe", uri)

        async def _disposer() -> None:
            cbs = self._subscriptions.get(uri)
            if cbs is None:
                return
            try:
                cbs.remove(callback)
            except ValueError:
                return
            if not cbs:
                # Last subscriber for this URI — release server-side too
                self._subscriptions.pop(uri, None)
                # P2-2: state == "running" alone is racy; gate on session
                # readiness instead. If session is reconnecting we skip
                # the unsubscribe (no-op is fine: subscriptions dict is
                # already empty for this uri so post-reconnect replay
                # won't resubscribe).
                if self._session_ready.is_set():
                    await self._send_command("unsubscribe", uri)

        return _disposer

    async def _send_command(
        self, op: Literal["subscribe", "unsubscribe"], uri: str,
    ) -> None:
        fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        await self._cmd_queue.put(_Cmd(op=op, uri=uri, future=fut))
        try:
            await fut
        except Exception as exc:
            _logger.warning(
                "mcp listener %r %s(%s) failed: %r",
                self._client.name, op, uri, exc,
            )
            # P1-1: subscribe failures propagate so callers know the
            # subscription is not actually active server-side. unsubscribe
            # failures are best-effort cleanup (server may have already
            # forgotten or transport is dying — silently log + continue).
            if op == "subscribe":
                raise

    # -----------------------------------------------------------------
    # Background task: connection loop with reconnect
    # -----------------------------------------------------------------

    async def _run(self) -> None:
        attempt = 0
        while True:
            try:
                await self._run_one_session()
                # _run_one_session returns normally only on shutdown.
                self._state = "stopped"
                return
            except StopReconnecting as exc:
                _logger.warning(
                    "mcp listener %r giving up: %r", self._client.name, exc,
                )
                self._state = "stopped_permanently"
                self._stop_reason = exc
                return
            except asyncio.CancelledError:
                self._state = "stopped"
                raise
            except Exception as exc:
                attempt += 1
                self._state = "reconnecting"
                self._session_ready.clear()
                _logger.warning(
                    "mcp listener %r session died (attempt %d): %r",
                    self._client.name, attempt, exc,
                )
                try:
                    delay = await self._reconnect.next_delay(attempt, exc)
                except StopReconnecting as stop:
                    _logger.warning(
                        "mcp listener %r reconnect strategy gave up: %r",
                        self._client.name, stop,
                    )
                    self._state = "stopped_permanently"
                    self._stop_reason = stop
                    return
                _logger.info(
                    "mcp listener %r reconnecting in %.1fs", self._client.name, delay,
                )
                await asyncio.sleep(delay)

    async def _run_one_session(self) -> None:
        """Establish one ClientSession lifecycle. Returns normally on
        shutdown command; raises on any other session error so `_run`
        can retry."""
        # Build a lightweight session_factory that injects our message_handler.
        # We can't reuse the client's session_factory directly because it
        # bakes in the per-call list_roots / sampling callbacks; the listener
        # needs a session with all of (list_roots / logging / sampling /
        # message_handler) attached so notifications + reverse RPCs both
        # work in this single long-lived connection.
        session_factory = _make_listener_session_factory(self._client, self)

        is_reconnect = self._first_session_done

        async with session_factory() as session:
            self._state = "running"
            self._session_ready.set()
            # Replay only on reconnect (P0-2 race fix: first-connect path
            # relies on the command queue; replay would double-call SDK).
            # Replay also re-validates cached lists are stale (P1-2):
            # server may have changed tools/prompts/resources during the
            # outage, the cache from before reconnect is no longer trusted.
            if is_reconnect:
                # Break on first failure (P1-3): if transport is dead,
                # subsequent uris will all fail and we'd block waiting for
                # commands on a dead session.
                for uri in list(self._subscriptions.keys()):
                    await self._sdk_subscribe(session, uri)
                # Invalidate caches so next list_tools/prompts/resources
                # hits the new session and pulls the latest catalog.
                for kind in ("tools", "prompts", "resources"):
                    try:
                        await self._client.notify_list_changed(kind)  # type: ignore[arg-type]
                    except Exception:
                        _logger.warning(
                            "mcp listener %r post-reconnect notify_list_changed(%s) "
                            "failed", self._client.name, kind, exc_info=True,
                        )
            self._first_session_done = True

            while True:
                cmd = await self._cmd_queue.get()
                if cmd.op == "shutdown":
                    cmd.future.set_result(None)
                    return
                try:
                    if cmd.op == "subscribe":
                        assert cmd.uri is not None
                        await self._sdk_subscribe(session, cmd.uri)
                    elif cmd.op == "unsubscribe":
                        assert cmd.uri is not None
                        await self._sdk_unsubscribe(session, cmd.uri)
                    cmd.future.set_result(None)
                except Exception as exc:
                    if not cmd.future.done():
                        cmd.future.set_exception(exc)
                    raise  # propagate to outer reconnect loop

    @staticmethod
    async def _sdk_subscribe(session: Any, uri: str) -> None:
        # MCP SDK accepts `pydantic.AnyUrl`; we pass str and let SDK coerce.
        await session.subscribe_resource(uri)

    @staticmethod
    async def _sdk_unsubscribe(session: Any, uri: str) -> None:
        await session.unsubscribe_resource(uri)

    # -----------------------------------------------------------------
    # Notification dispatch (called from message_handler in session task)
    # -----------------------------------------------------------------

    async def _on_notification(self, notification: Any) -> None:
        """Routes server notifications to the right consumer.

        Lazy import of `mcp.types` so module-level import doesn't pull
        the SDK. Unknown notification types log DEBUG and pass through
        — adding new MCP notification kinds shouldn't break the listener.
        """
        mcp_types = importlib.import_module("mcp.types")

        # `ServerNotification` is a discriminated union; the SDK exposes
        # the underlying notification at `notification.root`.
        inner = getattr(notification, "root", notification)

        if isinstance(inner, mcp_types.ResourceUpdatedNotification):
            uri = str(getattr(inner.params, "uri", ""))
            await self._dispatch_resource_updated(uri)
            return

        if isinstance(inner, mcp_types.ResourceListChangedNotification):
            await self._client.notify_list_changed("resources")
            return
        if isinstance(inner, mcp_types.ToolListChangedNotification):
            await self._client.notify_list_changed("tools")
            return
        if isinstance(inner, mcp_types.PromptListChangedNotification):
            await self._client.notify_list_changed("prompts")
            return

        _logger.debug(
            "mcp listener %r unhandled notification: %s",
            self._client.name, type(inner).__name__,
        )

    async def _dispatch_resource_updated(self, uri: str) -> None:
        callbacks = list(self._subscriptions.get(uri) or [])
        if not callbacks:
            return

        async def _run_one(cb: ResourceUpdatedCallback) -> None:
            try:
                result = cb(uri)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                _logger.warning(
                    "mcp listener %r callback for %s raised; "
                    "isolating other callbacks",
                    self._client.name, uri, exc_info=True,
                )

        # P2-1: run callbacks concurrently — slow callback no longer
        # head-of-lines other subscribers' notification dispatch.
        # Exceptions are caught inside _run_one so gather never raises.
        await asyncio.gather(*(_run_one(cb) for cb in callbacks))


# ---------------------------------------------------------------------------
# session_factory builder for the listener (separate from client's per-call)
# ---------------------------------------------------------------------------


def _make_listener_session_factory(
    client: "MCPClient", listener: MCPListener,
) -> Any:
    """Build an async-context-manager factory that opens a ClientSession
    with `message_handler` attached (so notifications flow into the
    listener), plus all the regular client capabilities (list_roots /
    logging / sampling).

    Re-uses the transport setup from `_make_real_session_factory_placeholder`
    by inspecting the client's MCPServerConfig stashed at construction
    time on `client._listener_config` (set by `MCPClient.from_config`
    so the listener can rebuild the same transport without re-parsing
    config).
    """
    config = getattr(client, "_listener_config", None)
    if config is None:
        raise RuntimeError(
            "MCPClient has no _listener_config; subscribe requires "
            "client constructed via MCPClient.from_config(...)"
        )

    @contextlib.asynccontextmanager
    async def factory() -> AsyncIterator[Any]:
        mcp_module = importlib.import_module("mcp")
        ClientSession = mcp_module.ClientSession

        async def message_handler(message: Any) -> None:
            # Filter exceptions and reverse-request responders; only
            # ServerNotification reaches our dispatcher. Reverse RPCs
            # (sampling, list_roots) are still handled by their own
            # dedicated callbacks (sampling_callback / list_roots_callback)
            # via SDK plumbing — message_handler doesn't substitute for
            # those.
            if isinstance(message, Exception):
                _logger.warning(
                    "mcp listener %r received exception via message_handler: %r",
                    client.name, message,
                )
                return
            # Detect ServerNotification by duck-typing `.root`
            if not hasattr(message, "root"):
                return
            try:
                await listener._on_notification(message)
            except Exception:
                _logger.warning(
                    "mcp listener %r notification dispatch failed",
                    client.name, exc_info=True,
                )

        list_roots_cb = (
            client._list_roots_callback
            if client._roots_provider is not None
            else None
        )
        sampling_cb = (
            client._sampling_callback
            if client._sampling_handler is not None
            else None
        )
        # Listener doesn't wrap the user logging callback — leave that to
        # the per-call session path; listening notifications.message
        # wouldn't be useful here anyway.

        from .types import MCPTransport

        if config.transport == MCPTransport.STDIO:
            stdio_mod = importlib.import_module("mcp.client.stdio")
            stdio_client = stdio_mod.stdio_client
            StdioServerParameters = mcp_module.StdioServerParameters

            server_params = StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env=dict(config.env) if config.env else None,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(
                    read, write,
                    message_handler=message_handler,
                    list_roots_callback=list_roots_cb,
                    sampling_callback=sampling_cb,
                ) as session:
                    await session.initialize()
                    yield session
            return

        if config.transport == MCPTransport.HTTP:
            http_mod = importlib.import_module("mcp.client.streamable_http")
            streamable_http_client = http_mod.streamable_http_client
            httpx_module = importlib.import_module("httpx")
            AsyncClient = httpx_module.AsyncClient

            async with AsyncClient(
                headers=config.headers or None,
                timeout=config.timeout,
                follow_redirects=False,
            ) as http_client:
                async with streamable_http_client(
                    url=config.url, http_client=http_client
                ) as (read, write):
                    async with ClientSession(
                        read, write,
                        message_handler=message_handler,
                        list_roots_callback=list_roots_cb,
                        sampling_callback=sampling_cb,
                    ) as session:
                        await session.initialize()
                        yield session
            return

        raise ValueError(f"unsupported MCP transport: {config.transport}")

    return factory


__all__ = [
    "ExponentialBackoff",
    "ListenerState",
    "MCPListener",
    "ReconnectStrategy",
    "ResourceUpdatedCallback",
    "StopReconnecting",
]
