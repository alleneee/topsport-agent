from __future__ import annotations

import contextlib
import importlib
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from .types import MCPServerConfig, MCPTransport

SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


class MCPClient:
    def __init__(self, name: str, session_factory: SessionFactory) -> None:
        self._name = name
        self._session_factory = session_factory
        self._cached_tools: list[Any] | None = None
        self._cached_prompts: list[Any] | None = None
        self._cached_resources: list[Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_config(cls, config: MCPServerConfig) -> MCPClient:
        return cls(config.name, _make_real_session_factory(config))

    async def list_tools(self, *, force_refresh: bool = False) -> list[Any]:
        if self._cached_tools is None or force_refresh:
            async with self._session_factory() as session:
                result = await session.list_tools()
                self._cached_tools = list(result.tools)
        return list(self._cached_tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        async with self._session_factory() as session:
            return await session.call_tool(name, arguments=arguments)

    async def list_prompts(self, *, force_refresh: bool = False) -> list[Any]:
        if self._cached_prompts is None or force_refresh:
            async with self._session_factory() as session:
                result = await session.list_prompts()
                self._cached_prompts = list(result.prompts)
        return list(self._cached_prompts)

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        async with self._session_factory() as session:
            return await session.get_prompt(name, arguments=arguments or {})

    async def list_resources(self, *, force_refresh: bool = False) -> list[Any]:
        if self._cached_resources is None or force_refresh:
            async with self._session_factory() as session:
                result = await session.list_resources()
                self._cached_resources = list(result.resources)
        return list(self._cached_resources)

    async def read_resource(self, uri: str) -> Any:
        async with self._session_factory() as session:
            return await session.read_resource(uri)

    def invalidate_cache(self) -> None:
        self._cached_tools = None
        self._cached_prompts = None
        self._cached_resources = None


def _make_real_session_factory(config: MCPServerConfig) -> SessionFactory:
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

        if config.transport == MCPTransport.HTTP:
            http_mod = importlib.import_module(http_module_name)
            streamable_http_client = http_mod.streamable_http_client
            httpx_module = importlib.import_module(httpx_module_name)
            AsyncClient = httpx_module.AsyncClient

            async with AsyncClient(
                headers=config.headers or None,
                timeout=config.timeout,
                follow_redirects=True,
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
