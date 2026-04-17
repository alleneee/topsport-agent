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
