from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MCPTransport(StrEnum):
    STDIO = "stdio"
    HTTP = "http"


@dataclass(slots=True)
class MCPServerConfig:
    """直接复用 Claude Desktop 的 JSON 配置结构，省去格式转换和迁移成本。"""
    name: str
    transport: MCPTransport
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    # Capability requirements contributed to every bridged ToolSpec from this
    # server. Empty means the MCP server's tools are visible to any session.
    permissions: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True, frozen=True)
class MCPToolInfo:
    """轻量描述一个 MCP 工具，便于测试 mock 和桥接层解耦真实 MCP SDK 对象。

    真实 MCP SDK 的 tool 对象通过 duck typing 暴露 `name` / `description` /
    `inputSchema`；`MCPToolInfo` 提供等价字段（`parameters` 即 inputSchema 别名）。
    桥接层对两种输入都能处理，因为 `_wrap` 用 getattr 按名查询。
    """
    name: str
    description: str = ""
    parameters: dict[str, object] = field(default_factory=lambda: {"type": "object"})

    @property
    def inputSchema(self) -> dict[str, object]:  # noqa: N802 — match MCP SDK 大小写
        """MCP SDK 原生字段名；桥接层走的是 inputSchema。"""
        return self.parameters
