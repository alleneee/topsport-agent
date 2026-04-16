from __future__ import annotations

from pathlib import Path

from .client import MCPClient
from .config import load_mcp_config
from .tool_bridge import MCPToolSource


class MCPManager:
    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    def register(self, client: MCPClient) -> None:
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
        return [MCPToolSource(client) for client in self._clients.values()]
