from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal

from .types import MCPServerConfig, MCPTransport

_logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]
ListKind = Literal["tools", "prompts", "resources"]
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
    ) -> None:
        self._name = name
        self._session_factory = session_factory
        self.permissions = permissions
        if cache_ttl is not None and cache_ttl < 0:
            raise ValueError(
                f"cache_ttl must be >= 0 or None, got {cache_ttl!r}"
            )
        self._cache_ttl = cache_ttl
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
        """生产入口；测试走 __init__ 直接注入 mock factory，不触碰真实 MCP 依赖。"""
        return cls(
            config.name,
            _make_real_session_factory(config),
            permissions=config.permissions,
            cache_ttl=config.cache_ttl,
        )

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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """写操作不走缓存，每次新建 session 保证 task 安全。"""
        async with self._session_factory() as session:
            return await session.call_tool(name, arguments=arguments)

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


def _make_real_session_factory(config: MCPServerConfig) -> SessionFactory:
    """通过变量间接 importlib.import_module 绕过 Pyright 的 reportMissingImports。

    mcp / httpx 都是可选依赖，只在 from_config 路径上触发导入。
    """
    mcp_module_name = "mcp"
    stdio_module_name = "mcp.client.stdio"
    http_module_name = "mcp.client.streamable_http"
    httpx_module_name = "httpx"

    @contextlib.asynccontextmanager
    async def factory() -> AsyncIterator[Any]:
        mcp_module = importlib.import_module(mcp_module_name)
        ClientSession = mcp_module.ClientSession

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
                async with ClientSession(read, write) as session:
                    await session.initialize()
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
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        yield session
            return

        raise ValueError(f"unsupported MCP transport: {config.transport}")

    return factory
