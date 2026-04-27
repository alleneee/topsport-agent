from __future__ import annotations

from pathlib import Path

from .client import MCPClient
from .config import load_mcp_config
from .roots import RootsProvider
from .tool_bridge import MCPToolSource


class MCPManager:
    """多服务编排器：统一管理所有 MCP 客户端，对引擎暴露聚合的 tool_sources 列表。"""
    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    def register(self, client: MCPClient, *, replace: bool = False) -> None:
        """注册 MCPClient。同名已存在时默认 raise（避免静默覆盖丢能力 + 旧 client
        资源泄漏）。`replace=True` 时显式允许替换，但 caller 自己负责关旧 client。
        """
        existing = self._clients.get(client.name)
        if existing is not None and not replace:
            raise ValueError(
                f"MCP client name {client.name!r} already registered; "
                f"pass replace=True if intentional"
            )
        self._clients[client.name] = client

    @classmethod
    def from_config_file(cls, path: str | Path) -> MCPManager:
        manager = cls()
        for config in load_mcp_config(path):
            manager.register(MCPClient.from_config(config))
        return manager

    def clients(self) -> list[MCPClient]:
        return list(self._clients.values())

    def get(self, name: str) -> MCPClient | None:
        return self._clients.get(name)

    def tool_sources(self) -> list[MCPToolSource]:
        """每次调用都新建 MCPToolSource，引擎每步重新快照工具列表时自然拿到最新状态。"""
        return [MCPToolSource(client) for client in self._clients.values()]

    def set_roots_provider(self, provider: RootsProvider | None) -> None:
        """Apply the same `RootsProvider` to every registered client.

        The provider is what MCP servers see when they call `roots/list`.
        Setting None clears the provider on every client (next session
        won't declare the `roots` capability). Subsequent `register()`
        calls do NOT inherit this provider — operators must set the
        provider after the manager is fully populated, or call
        set_roots_provider again.
        """
        for client in self._clients.values():
            client.set_roots_provider(provider)
