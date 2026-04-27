from __future__ import annotations

import contextlib
import importlib
import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal

from . import elicitation as _elicit
from .elicitation import ElicitationHandler
from .logging_handler import LoggingCallback, MCPLogLevel
from .progress import ProgressCallback, wrap_progress_callback
from .roots import Root, RootsProvider, call_roots_provider
from .sampling import (
    SamplingHandler,
    from_sdk_params,
    to_sdk_error,
    to_sdk_result,
)
from .types import MCPServerConfig, MCPTransport

_logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]
ListKind = Literal["tools", "prompts", "resources"]


class _Unset:
    """Sentinel type for `call_tool(progress_callback=...)`: distinguishes
    "param not passed → use client default" from "explicit None → disable
    for this call". Singleton instance `_UNSET` is the only intended value.
    """


_UNSET = _Unset()
# 回调可同步可异步：listening session 触发时是 async 上下文，应用层手动
# notify 也兼容；inspect.iscoroutine 在 dispatch 时分流。
ListChangedCallback = Callable[[ListKind], "Awaitable[None] | None"]
Disposer = Callable[[], None]


class MCPClient:
    """MCP 会话不能跨 asyncio task 共享（cancel scope 绑定到创建它的 task）。

    因此每次调用都新建一个短生命周期的 session；列表结果通过带 TTL 的缓存
    复用，过期或显式 force_refresh / invalidate_cache / notify_list_changed
    时拉取最新。完整的 listChanged 通知订阅需要 long-lived listening session
    （架构调整），见 `subscribe_list_changed` 的 follow-up 说明。

    Cache TTL 语义见 MCPServerConfig.cache_ttl 字段注释：
      None=永不过期，0=不缓存（每次拉），正数=过期秒数，负数 raise。
    """
    def __init__(
        self,
        name: str,
        session_factory: SessionFactory,
        *,
        permissions: frozenset[str] = frozenset(),
        cache_ttl: float | None = 60.0,
        clock: Callable[[], float] = time.monotonic,
        roots_provider: RootsProvider | None = None,
    ) -> None:
        self._name = name
        self._session_factory = session_factory
        self.permissions = permissions
        if cache_ttl is not None and cache_ttl < 0:
            raise ValueError(
                f"cache_ttl must be >= 0 or None, got {cache_ttl!r}"
            )
        self._cache_ttl = cache_ttl
        # roots_provider: server-side fs root advertisement. None means client
        # does NOT declare the `roots` capability (legacy behavior). Set to a
        # provider via `set_roots_provider` or pass at construction time; the
        # session_factory below converts it into the SDK's list_roots_callback.
        self._roots_provider = roots_provider
        # logging_callback: receives server's notifications/message events.
        # None = no callback registered (server still sends notifications but
        # ClientSession discards them). _logging_level is sent via
        # session.set_logging_level(...) right after initialize when set.
        self._logging_callback: LoggingCallback | None = None
        self._logging_level: MCPLogLevel | None = None
        # progress_callback: per-call hook injected when the bridged
        # MCPToolSource invokes call_tool. None disables the capability
        # (no `_meta.progressToken` sent → server cannot emit progress).
        self._progress_callback: ProgressCallback | None = None
        # sampling_handler: server-driven LLM call resolver. None means
        # client does NOT declare the `sampling` capability (legacy + safe
        # default — server can't co-opt operator's LLM credit).
        self._sampling_handler: SamplingHandler | None = None
        # MCPListener for resources/subscribe + list_changed notifications
        # (lazy-created by subscribe_resource). Requires _listener_config
        # to be set by `from_config`; bypassing from_config (test mocks)
        # leaves both as None and subscription is unavailable.
        self._listener: Any = None
        self._listener_config: MCPServerConfig | None = None
        # elicitation_handler: server-driven user-input request resolver.
        # None means client does NOT declare the `elicitation` capability
        # (server can't ask the user for input). Default safe.
        self._elicitation_handler: ElicitationHandler | None = None
        # Current tool-call session_id (set by MCPToolSource handler), read
        # by `_elicitation_callback` to route server-initiated elicitation
        # back to the right user. **Cannot use ContextVar**: SDK message
        # handler runs in its own task whose context snapshot was taken
        # at session start (before any tool call), so ContextVar never
        # reaches it. Instance field + lock is the correct cross-task
        # shared-state primitive at the cost of serialising tool calls
        # **only when elicitation is enabled** (operators opt into the
        # serialisation by enabling the capability).
        self._current_call_session_id: str | None = None
        self._call_lock = asyncio.Lock()
        # clock contract: 必须单调递增。生产默认 time.monotonic（避免 wall-clock
        # 跳变扰动 TTL 判定）；测试可注入可控函数。如果调用方误传非单调函数
        # （如 time.time）导致 age 计算出负值，_is_fresh 会按"已过期"处理（保守
        # 兜底），下次访问自动 refresh。
        self._clock = clock
        # 缓存：(value, populated_at)；populated_at 是 self._clock() 取得的时间戳。
        self._cache_tools: tuple[list[Any], float] | None = None
        self._cache_prompts: tuple[list[Any], float] | None = None
        self._cache_resources: tuple[list[Any], float] | None = None
        # listChanged 回调：架构占位，long-lived listening session 实装时由其触发。
        self._list_changed_callbacks: list[ListChangedCallback] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def cache_ttl(self) -> float | None:
        return self._cache_ttl

    @classmethod
    def from_config(cls, config: MCPServerConfig) -> MCPClient:
        """生产入口；测试走 __init__ 直接注入 mock factory，不触碰真实 MCP 依赖。

        Roots provider 是运行时绑定（不在 server config 文件里），通过
        `set_roots_provider` 设置；这里构造的 client 默认无 roots 声明。
        """
        # NOTE on the placeholder dance: 真正的 session_factory 闭包需要 client
        # 引用（用于 list_roots_callback dispatch 到当前的 _roots_provider），
        # 而 client 又需要一个 session_factory 才能构造。两步必须**紧贴执行**：
        # 不要在 cls(...) 与 swap 之间插入任何会 await 或可能调 list_tools/_call
        # 等访问 _session_factory 的逻辑——placeholder 被调用即抛 RuntimeError。
        client = cls(
            config.name,
            _make_real_session_factory_placeholder(config),
            permissions=config.permissions,
            cache_ttl=config.cache_ttl,
        )
        client._session_factory = _make_real_session_factory(config, client)
        # Stash a deep-frozen snapshot of the config (P2-3 review): if
        # callers mutate `config.env` etc. after `from_config`, the
        # listener should NOT silently pick up those changes on next
        # reconnect — that's a debug nightmare. dataclasses.replace is
        # a shallow copy, paired with the dataclass being declared
        # frozen on the relevant fields keeps the snapshot stable.
        import dataclasses as _dc
        client._listener_config = _dc.replace(config)
        return client

    # -----------------------------------------------------------------
    # Roots capability (client → server fs boundary advertisement)
    # -----------------------------------------------------------------

    def set_roots_provider(self, provider: RootsProvider | None) -> None:
        """Replace the active roots provider. None disables the capability
        (the next initialize handshake will not declare `roots`).

        Limitation: this currently does NOT emit the spec's
        `notifications/roots/list_changed`. Servers that issue a fresh
        `roots/list` (per-session, after each connect) see the new
        provider; servers that cached results from an earlier session
        won't refresh until reconnect. Tied to the Phase 5.1 listening-
        session follow-up — once a long-lived session exists, this
        method should also push the change notification."""
        self._roots_provider = provider

    @property
    def roots_provider(self) -> RootsProvider | None:
        return self._roots_provider

    # -----------------------------------------------------------------
    # Logging capability (server -> client message stream)
    # -----------------------------------------------------------------

    def set_logging_callback(
        self,
        callback: LoggingCallback | None,
        *,
        level: MCPLogLevel | None = None,
    ) -> None:
        """Register a callback for `notifications/message` events.

        `callback`: invoked for every server log message
        (`(LoggingMessageNotificationParams) -> Awaitable[None]`). The
        session_factory wraps it in an exception-isolating shim so a
        bad callback can't crash the SDK's notification loop.
        `level`: when non-None, the next session's `initialize` is
        followed by `logging/setLevel` to filter at the server side.
        Pass None to leave the threshold at the server's default.

        Setting `callback=None` disables the capability for subsequent
        sessions; the threshold stays as last-set until cleared.

        Rationale (callback / level decoupling): operators commonly
        toggle the callback for temporary muting (debug pause / sampling)
        and want to re-enable without rewiring level. To clear both,
        call `set_logging_callback(None)` and `set_logging_level(None)`
        in sequence. Server may keep emitting at the previous level
        when only the callback is None — wastes a few bytes but is
        intentional.
        """
        self._logging_callback = callback
        if level is not None:
            self._logging_level = level

    # -----------------------------------------------------------------
    # Resource subscription (long-lived listening session)
    # -----------------------------------------------------------------

    async def subscribe_resource(
        self,
        uri: str,
        callback: Callable[[str], Any],
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe to `notifications/resources/updated` for `uri`.

        Returns an async disposer; awaiting unsubscribes (decrements
        refcount, releases server-side subscription when last subscriber
        for the URI removes itself).

        Lazy: first call constructs the long-lived `MCPListener` and
        starts the listening session. Reconnect on session failure is
        automatic via `ExponentialBackoff` strategy by default. The
        returned disposer is idempotent — calling it twice is safe.

        Server pushes only the URI on update; callbacks must call
        `read_resource(uri)` themselves to fetch the new content (MCP
        spec hint-then-pull design).
        """
        if self._listener_config is None:
            raise RuntimeError(
                "MCPClient.subscribe_resource requires construction via "
                "MCPClient.from_config(...); test fixtures using __init__ "
                "directly cannot use this capability"
            )
        if self._listener is None:
            from .listener import MCPListener
            self._listener = MCPListener(self)
        return await self._listener.subscribe_resource(uri, callback)

    async def close(self) -> None:
        """Stop the listener (if running) and release any background
        resources. Safe to call multiple times. Per-call short-lived
        sessions don't need explicit close — they're context-managed."""
        if self._listener is not None:
            try:
                await self._listener.stop()
            except Exception:
                _logger.warning(
                    "mcp client %r listener stop failed", self._name,
                    exc_info=True,
                )

    # -----------------------------------------------------------------
    # Sampling capability (server -> client LLM-call advertisement)
    # -----------------------------------------------------------------

    def set_sampling_handler(self, handler: SamplingHandler | None) -> None:
        """Register a handler for `sampling/createMessage` requests.

        Setting this declares the `sampling` capability — server can
        thereafter ask the client to make LLM calls on its behalf.
        Setting None disables (default; the client does NOT advertise
        the capability and server requests will return method-not-found).

        Security note: enabling sampling lends operator's LLM credit to
        whatever MCP server is connected. Always pair with a token cap
        (`LLMProviderSamplingHandler.max_tokens_cap`) and an allowlist
        of trusted servers / models.
        """
        self._sampling_handler = handler

    @property
    def sampling_handler(self) -> SamplingHandler | None:
        return self._sampling_handler

    # -----------------------------------------------------------------
    # Elicitation capability (server -> client user-input request)
    # -----------------------------------------------------------------

    def set_elicitation_handler(
        self, handler: ElicitationHandler | None,
    ) -> None:
        """Register a handler for `elicitation/create` requests.

        Setting this declares the `elicitation` capability — server can
        thereafter ask the user for input (form schema or external URL
        confirmation). Setting None disables (default; server requests
        return method-not-found).

        UX note: handler is invoked from the SDK's message-handling
        task; production HTTP server impl uses contextvars to route
        the request to the user session that triggered the underlying
        tool call (see `mcp.elicitation.current_session_id` ContextVar)."""
        self._elicitation_handler = handler

    @property
    def elicitation_handler(self) -> ElicitationHandler | None:
        return self._elicitation_handler

    async def _elicitation_callback(self, _context: Any, params: Any) -> Any:
        """Adapter from our ElicitationHandler → MCP SDK's ElicitationFnT.

        Reads `_current_call_session_id` from the client instance (set by
        MCPToolSource handler under `_call_lock`), so routing works
        across the SDK task boundary where ContextVar fails."""
        import uuid as _uuid

        handler = self._elicitation_handler
        if handler is None:
            return _elicit.to_sdk_error(
                "client did not register an elicitation handler",
                code=-32601,  # Method not found
            )
        try:
            request = _elicit.from_sdk_params(
                params,
                request_id=_uuid.uuid4().hex,
                session_id=self._current_call_session_id,
            )
        except Exception as exc:
            _logger.warning(
                "mcp client %r failed to parse elicitation params: %r",
                self._name, exc, exc_info=True,
            )
            return _elicit.to_sdk_error(
                f"invalid elicitation params: {type(exc).__name__}: {exc}",
                code=-32602,
            )
        try:
            response = await handler(request)
        except Exception as exc:
            _logger.warning(
                "mcp client %r elicitation_handler raised %r",
                self._name, exc, exc_info=True,
            )
            return _elicit.to_sdk_error(
                f"elicitation handler failed: {type(exc).__name__}: {exc}",
            )
        return _elicit.to_sdk_result(response)

    async def _sampling_callback(self, _context: Any, params: Any) -> Any:
        """Adapter from our SamplingHandler → MCP SDK's SamplingFnT.

        Lazy import of `mcp.types` happens inside `from_sdk_params` /
        `to_sdk_result` / `to_sdk_error` so test code with mocked SDK
        doesn't need the real package. Errors from the handler turn into
        `ErrorData` instead of raising — raising would deadlock the
        server's createMessage call."""
        handler = self._sampling_handler
        if handler is None:
            return to_sdk_error(
                "client did not register a sampling handler",
                code=-32601,  # Method not found
            )
        try:
            request = from_sdk_params(params)
        except Exception as exc:
            _logger.warning(
                "mcp client %r failed to parse sampling params: %r",
                self._name, exc, exc_info=True,
            )
            return to_sdk_error(
                f"invalid sampling params: {type(exc).__name__}: {exc}",
                code=-32602,  # Invalid params
            )
        from .sampling import RateLimitExceeded
        try:
            result = await handler(request)
        except RateLimitExceeded as exc:
            # 单独 code 让 server 知道是速率限制可退避重试，而不是 handler 故障。
            # JSON-RPC 规范保留 -32000..-32099 作为 server-defined errors；
            # MCP 没标准化 rate-limit code，这里选 -32000 作为该范围首位。
            _logger.warning(
                "mcp client %r sampling rate limit exceeded: %s",
                self._name, exc,
            )
            return to_sdk_error(
                f"rate limit exceeded: {exc}", code=-32000,
            )
        except Exception as exc:
            _logger.warning(
                "mcp client %r sampling_handler raised %r",
                self._name, exc, exc_info=True,
            )
            return to_sdk_error(
                f"sampling handler failed: {type(exc).__name__}: {exc}",
            )
        return to_sdk_result(result)

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Register a progress callback used for every `call_tool` invocation.

        Per MCP spec the callback is per-call (each request opts into
        progress via its own `_meta.progressToken`). Setting it here is
        a convenience: subsequent call_tool calls automatically pass the
        callback, so MCPToolSource's auto-bridged tools get progress
        reporting without each call site wiring it up.

        None disables progress reporting (no progressToken sent → server
        cannot emit notifications/progress)."""
        self._progress_callback = callback

    @property
    def progress_callback(self) -> ProgressCallback | None:
        return self._progress_callback

    def set_logging_level(self, level: MCPLogLevel | None) -> None:
        """Set the level threshold sent via `logging/setLevel` after
        each subsequent initialize. Independent from
        `set_logging_callback` so operators can adjust verbosity without
        touching the callback chain."""
        self._logging_level = level

    @property
    def logging_callback(self) -> LoggingCallback | None:
        return self._logging_callback

    @property
    def logging_level(self) -> MCPLogLevel | None:
        return self._logging_level

    async def _list_roots_callback(self, _context: Any) -> Any:
        """Adapter from our RootsProvider → MCP SDK's ListRootsFnT.

        Imports MCP SDK types lazily so test code that mocks session_factory
        can avoid pulling the SDK. SDK errors and provider errors both turn
        into ErrorData responses to the server (per spec; raising here would
        deadlock the server's roots/list call)."""
        provider = self._roots_provider
        if provider is None:
            mcp_types = importlib.import_module("mcp.types")
            return mcp_types.ErrorData(
                code=-32601,  # Method not found
                message="client did not register a roots provider",
            )
        try:
            roots: list[Root] = await call_roots_provider(provider)
        except Exception as exc:
            mcp_types = importlib.import_module("mcp.types")
            _logger.warning(
                "mcp client %r roots_provider raised %r", self._name, exc,
                exc_info=True,
            )
            return mcp_types.ErrorData(
                code=-32603,  # Internal error
                message=f"roots provider failed: {type(exc).__name__}: {exc}",
            )
        return _to_list_roots_result(roots)

    # -----------------------------------------------------------------
    # listChanged 订阅
    # -----------------------------------------------------------------

    def subscribe_list_changed(
        self, callback: ListChangedCallback,
    ) -> Disposer:
        """注册 list_changed 回调，返回 disposer（调用即注销）。

        Callback 签名 `(kind) -> Awaitable[None] | None`：sync 和 async 都接。
        ⚠️ 当前 transport 不主动触发该路径（短生命周期 session 听不到 server
        的 notifications/list_changed）；架构占位，follow-up：
          1. 给 MCPClient 加 background task 跑 long-lived listening session
          2. 把 server 推来的 list_changed dispatch 到 `notify_list_changed`
        应用层（已知 server 重启等外部信号时）也可主动调 `notify_list_changed`。
        """
        self._list_changed_callbacks.append(callback)

        def _dispose() -> None:
            try:
                self._list_changed_callbacks.remove(callback)
            except ValueError:
                # 已注销 / 重复注销 — 幂等
                pass

        return _dispose

    async def notify_list_changed(self, kind: ListKind) -> None:
        """触发 kind 对应的缓存失效 + 串行调用所有订阅者回调。

        - 缓存先失效，后跑回调，确保回调内若调 list_*() 拿到最新值。
        - sync 回调直接调；async 回调 await。
        - 单个回调抛异常 → log warning 隔离，不中断回调链。
        - 公开 API：listening session task 与应用层均可调用。
        """
        if kind == "tools":
            self._cache_tools = None
        elif kind == "prompts":
            self._cache_prompts = None
        elif kind == "resources":
            self._cache_resources = None
        # 快照回调列表，避免回调内 unsubscribe 引发的并发改动
        for cb in list(self._list_changed_callbacks):
            try:
                result = cb(kind)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                _logger.warning(
                    "mcp client %r list_changed callback failed for kind=%s",
                    self._name, kind, exc_info=True,
                )

    # -----------------------------------------------------------------
    # List operations with TTL caching
    # -----------------------------------------------------------------

    def _is_fresh(self, entry: tuple[list[Any], float] | None) -> bool:
        if entry is None:
            return False
        ttl = self._cache_ttl
        if ttl is None:
            return True  # 永不过期
        if ttl == 0:
            return False  # 不缓存：每次都拉
        age = self._clock() - entry[1]
        if age < 0:
            # 时钟回拨（非 monotonic clock 注入）— 保守视为 stale
            return False
        return age < ttl

    async def _load_tools(self) -> list[Any]:
        async with self._session_factory() as session:
            result = await session.list_tools()
            return list(result.tools)

    async def _load_prompts(self) -> list[Any]:
        async with self._session_factory() as session:
            result = await session.list_prompts()
            return list(result.prompts)

    async def _load_resources(self) -> list[Any]:
        async with self._session_factory() as session:
            result = await session.list_resources()
            return list(result.resources)

    async def list_tools(self, *, force_refresh: bool = False) -> list[Any]:
        """返回工具列表。返回的 list 是缓存的浅拷贝；列表元素仍是缓存里的
        同一对象，调用方不得 mutate 元素字段（与 MCP SDK 返回对象 immutable
        约定一致）。"""
        cache = self._cache_tools
        if force_refresh or not self._is_fresh(cache):
            tools = await self._load_tools()
            self._cache_tools = (tools, self._clock())
            return list(tools)
        # cache 必非 None：上面 _is_fresh(None) → False，已走 refresh 分支
        # （用本地变量替代 assert，兼容 python -O）
        return list(cache[0]) if cache is not None else []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        progress_callback: ProgressCallback | None | _Unset = _UNSET,
    ) -> Any:
        """写操作不走缓存，每次新建 session 保证 task 安全。

        `progress_callback` 三态语义（避免 Phase 4 抓过的"None 被 or 默认值覆盖"
        模式 + Phase 5.1 的"None 显式有意义"约定）：
          - 不传（默认 _UNSET）：使用 client._progress_callback（即
            `set_progress_callback` 设的全局 default）。
          - 显式 None：本次禁用 progress（即便 client default 已设）。
          - 非 None ProgressCallback：本次专用，覆盖 client default。
        最终生效的 callback 由 wrap_progress_callback 包一层做异常隔离。
        """
        if isinstance(progress_callback, _Unset):
            effective_cb = self._progress_callback
        else:
            effective_cb = progress_callback  # 含显式 None
        sdk_cb = (
            wrap_progress_callback(effective_cb, client_name=self._name)
            if effective_cb is not None else None
        )
        async with self._session_factory() as session:
            kwargs: dict[str, Any] = {"arguments": arguments}
            if sdk_cb is not None:
                kwargs["progress_callback"] = sdk_cb
            return await session.call_tool(name, **kwargs)

    async def list_prompts(self, *, force_refresh: bool = False) -> list[Any]:
        """返回 prompt 列表（浅拷贝；同 list_tools 注释）。"""
        cache = self._cache_prompts
        if force_refresh or not self._is_fresh(cache):
            prompts = await self._load_prompts()
            self._cache_prompts = (prompts, self._clock())
            return list(prompts)
        return list(cache[0]) if cache is not None else []

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        async with self._session_factory() as session:
            return await session.get_prompt(name, arguments=arguments or {})

    async def list_resources(self, *, force_refresh: bool = False) -> list[Any]:
        """返回 resource 列表（浅拷贝；同 list_tools 注释）。"""
        cache = self._cache_resources
        if force_refresh or not self._is_fresh(cache):
            resources = await self._load_resources()
            self._cache_resources = (resources, self._clock())
            return list(resources)
        return list(cache[0]) if cache is not None else []

    async def read_resource(self, uri: str) -> Any:
        async with self._session_factory() as session:
            return await session.read_resource(uri)

    def invalidate_cache(self) -> None:
        self._cache_tools = None
        self._cache_prompts = None
        self._cache_resources = None


def _make_real_session_factory_placeholder(
    config: MCPServerConfig,
) -> SessionFactory:
    """Boot-time placeholder; replaced inside `from_config` once the client
    instance exists (so `list_roots_callback` can dispatch into the live
    `_roots_provider`)."""
    del config

    @contextlib.asynccontextmanager
    async def _placeholder() -> AsyncIterator[Any]:
        raise RuntimeError(
            "MCPClient session_factory accessed before from_config bound the "
            "real factory; this is a programming error"
        )
        yield  # pragma: no cover  (unreachable, keeps generator type)

    return _placeholder


def _make_real_session_factory(
    config: MCPServerConfig,
    client: "MCPClient",
) -> SessionFactory:
    """通过变量间接 importlib.import_module 绕过 Pyright 的 reportMissingImports。

    mcp / httpx 都是可选依赖，只在 from_config 路径上触发导入。
    `client` reference is captured so the per-session list_roots_callback
    dispatches through the client's *current* roots_provider — operators
    can swap the provider after registration and the next session reflects
    it (no need to rebuild the manager).
    """
    mcp_module_name = "mcp"
    stdio_module_name = "mcp.client.stdio"
    http_module_name = "mcp.client.streamable_http"
    httpx_module_name = "httpx"

    @contextlib.asynccontextmanager
    async def factory() -> AsyncIterator[Any]:
        mcp_module = importlib.import_module(mcp_module_name)
        ClientSession = mcp_module.ClientSession

        # Only attach the callback when a provider is set; otherwise leave
        # ClientSession's default behavior (roots capability not declared).
        list_roots_cb = (
            client._list_roots_callback
            if client._roots_provider is not None
            else None
        )
        # Same conditional attach for sampling: if no handler set, don't
        # declare the capability (server gets method-not-found if it tries).
        sampling_cb = (
            client._sampling_callback
            if client._sampling_handler is not None
            else None
        )
        # Elicitation: same conditional pattern.
        elicitation_cb = (
            client._elicitation_callback
            if client._elicitation_handler is not None
            else None
        )
        # Logging callback: wrap user callback in a try/except so a bad
        # callback can't bring down the SDK's notification dispatch loop
        # (mirrors notify_list_changed exception isolation).
        user_logging_cb = client._logging_callback
        if user_logging_cb is not None:
            async def _safe_logging_cb(params: Any) -> None:
                try:
                    await user_logging_cb(params)
                except Exception:
                    _logger.warning(
                        "mcp client %r logging_callback raised; dropping message",
                        client._name, exc_info=True,
                    )
            logging_cb: Any = _safe_logging_cb
        else:
            logging_cb = None
        # If a level was set, send `logging/setLevel` right after initialize so
        # the server can pre-filter (avoids cluttering callback chain with
        # debug spam when operator only wants WARNING+).
        logging_level = client._logging_level

        async def _post_init(session: Any) -> None:
            if logging_level is None:
                return
            setlevel = getattr(session, "set_logging_level", None)
            if setlevel is None:
                # SDK 不支持（旧版本）—— server-side filter 保持默认级别。
                _logger.info(
                    "mcp client %r SDK lacks set_logging_level; "
                    "server will use its default verbosity",
                    client._name,
                )
                return
            try:
                await setlevel(logging_level)
            except AttributeError:
                # 防御：getattr 已挡，万一 setlevel 内部 AttributeError
                _logger.info(
                    "mcp client %r set_logging_level not callable",
                    client._name,
                )
            except Exception as exc:
                # Server protocol error (拒绝级别 / RPC 失败) - separate from
                # SDK absence so ops alerting can distinguish.
                _logger.warning(
                    "mcp client %r server rejected set_logging_level(%s): %r",
                    client._name, logging_level, exc, exc_info=True,
                )

        if config.transport == MCPTransport.STDIO:
            stdio_mod = importlib.import_module(stdio_module_name)
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
                    list_roots_callback=list_roots_cb,
                    logging_callback=logging_cb,
                    sampling_callback=sampling_cb,
                    elicitation_callback=elicitation_cb,
                ) as session:
                    await session.initialize()
                    await _post_init(session)
                    yield session
            return

        # HTTP 传输走 MCP SDK v2 的 streamable_http_client，外部 httpx 客户端负责 headers/timeout。
        if config.transport == MCPTransport.HTTP:
            http_mod = importlib.import_module(http_module_name)
            streamable_http_client = http_mod.streamable_http_client
            httpx_module = importlib.import_module(httpx_module_name)
            AsyncClient = httpx_module.AsyncClient

            # H-S1: follow_redirects=False 防 SSRF/metadata 泄露。
            # 若 MCP server 回 3xx，MCP SDK 会收到重定向响应并自行决定处理；
            # 这杜绝 httpx 自动把 headers（含 Authorization）重放给重定向目标。
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
                        list_roots_callback=list_roots_cb,
                        logging_callback=logging_cb,
                        sampling_callback=sampling_cb,
                        elicitation_callback=elicitation_cb,
                    ) as session:
                        await session.initialize()
                        await _post_init(session)
                        yield session
            return

        raise ValueError(f"unsupported MCP transport: {config.transport}")

    return factory


def _to_list_roots_result(roots: list[Root]) -> Any:
    """Convert our `Root` dataclass list into MCP SDK's `ListRootsResult`.

    Lazy import keeps mcp SDK out of the import path of callers that don't
    use real transports (e.g. tests with mock session factories).
    """
    mcp_types = importlib.import_module("mcp.types")
    sdk_roots = []
    for r in roots:
        kwargs: dict[str, Any] = {"uri": r.uri}
        if r.name is not None:
            kwargs["name"] = r.name
        if r.meta is not None:
            kwargs["meta"] = r.meta
        sdk_roots.append(mcp_types.Root(**kwargs))
    return mcp_types.ListRootsResult(roots=sdk_roots)
