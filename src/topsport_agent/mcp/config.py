from __future__ import annotations

import json
from pathlib import Path

from .types import MCPServerConfig, MCPTransport


def load_mcp_config(path: str | Path) -> list[MCPServerConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_servers = data.get("mcpServers", {})
    if not isinstance(raw_servers, dict):
        raise ValueError("mcp config: 'mcpServers' must be an object")

    configs: list[MCPServerConfig] = []
    for name, raw in raw_servers.items():
        if not isinstance(raw, dict):
            raise ValueError(f"mcp config: server '{name}' must be an object")
        transport_value = raw.get("transport", "stdio")
        try:
            transport = MCPTransport(transport_value)
        except ValueError as exc:
            raise ValueError(
                f"mcp config: server '{name}' has unknown transport '{transport_value}'"
            ) from exc

        config = MCPServerConfig(
            name=name,
            transport=transport,
            command=raw.get("command"),
            args=list(raw.get("args", [])),
            env=dict(raw.get("env", {})),
            url=raw.get("url"),
            headers=dict(raw.get("headers", {})),
            timeout=float(raw.get("timeout", 30.0)),
        )
        _validate(config)
        configs.append(config)
    return configs


def _validate(config: MCPServerConfig) -> None:
    if config.transport == MCPTransport.STDIO:
        if not config.command:
            raise ValueError(
                f"mcp config: server '{config.name}' stdio requires 'command'"
            )
    elif config.transport == MCPTransport.HTTP:
        if not config.url:
            raise ValueError(
                f"mcp config: server '{config.name}' http requires 'url'"
            )
