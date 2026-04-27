from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .policy import AllowEntry, MCPSecurityPolicy, enforce_stdio_policy
from .types import MCPServerConfig, MCPTransport


def load_mcp_config(
    path: str | Path,
    *,
    policy: MCPSecurityPolicy | None = None,
) -> list[MCPServerConfig]:
    """读取 Claude Desktop 格式的 mcpServers JSON 并校验 stdio 安全策略。

    策略解析顺序：
    1. 调用方显式传入的 policy 优先
    2. JSON 文件顶层存在 `allowlist` 字段 → 自动 strict
    3. 否则 permissive（兼容历史配置，但日志会对可疑 stdio 启动记告警）
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_servers = data.get("mcpServers", {})
    if not isinstance(raw_servers, dict):
        raise ValueError("mcp config: 'mcpServers' must be an object")

    effective_policy = policy or _policy_from_data(data)

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
            permissions=frozenset(raw.get("permissions", [])),
            cache_ttl=_parse_cache_ttl(raw.get("cache_ttl", 60.0)),
        )
        _validate(config)
        if config.transport == MCPTransport.STDIO:
            enforce_stdio_policy(
                server_name=config.name,
                command=config.command,
                args=list(config.args),
                policy=effective_policy,
            )
        configs.append(config)
    return configs


def _policy_from_data(data: dict) -> MCPSecurityPolicy:
    """配置文件顶层可选 `allowlist: [{name, command, args_prefix?}, ...]`。
    出现即 strict；不出现 permissive 以兼容历史配置。
    """
    raw_allowlist = data.get("allowlist")
    if raw_allowlist is None:
        return MCPSecurityPolicy.permissive()
    if not isinstance(raw_allowlist, list):
        raise ValueError("mcp config: 'allowlist' must be an array")

    entries: list[AllowEntry] = []
    for idx, raw in enumerate(raw_allowlist):
        if not isinstance(raw, dict):
            raise ValueError(
                f"mcp config: allowlist[{idx}] must be an object"
            )
        try:
            name = str(raw["name"])
            command = str(raw["command"])
        except KeyError as exc:
            raise ValueError(
                f"mcp config: allowlist[{idx}] missing required field {exc}"
            ) from exc
        args_prefix = tuple(str(a) for a in raw.get("args_prefix", []))
        entries.append(AllowEntry(name=name, command=command, args_prefix=args_prefix))
    return MCPSecurityPolicy.strict(entries)


def _parse_cache_ttl(raw: Any) -> float | None:
    """JSON config 解析：None / null → None（永不过期）；数值转 float。"""
    if raw is None:
        return None
    return float(raw)


def _validate(config: MCPServerConfig) -> None:
    """stdio 必须有 command，http 必须有 url -- 传输层缺少入口点就无法建立连接。"""
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
