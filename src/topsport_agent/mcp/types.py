from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MCPTransport(StrEnum):
    STDIO = "stdio"
    HTTP = "http"


@dataclass(slots=True)
class MCPServerConfig:
    """直接复用 Claude Desktop 的 JSON 配置结构，省去格式转换和迁移成本。

    敏感字段 (`env`, `headers`) 用 `repr=False`：env 常含 BRAVE_API_KEY 等
    第三方凭证、headers 常含 `Authorization: Bearer <token>`，dataclass 默认
    repr 会让 `logger.debug("registering %r", cfg)` / FastAPI debug traceback
    把这些直接 dump 到日志或前端。
    """
    name: str
    transport: MCPTransport
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict, repr=False)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    timeout: float = 30.0
    # Capability requirements contributed to every bridged ToolSpec from this
    # server. Empty means the MCP server's tools are visible to any session.
    permissions: frozenset[str] = field(default_factory=frozenset)
    # 缓存 list_tools / list_prompts / list_resources 结果的最长秒数。
    # 三种取值语义：
    #   None  —— 永不过期，仅显式 invalidate / force_refresh / list_changed
    #            通知触发刷新（适合工具集稳定的长跑场景）
    #   0     —— 立即过期 / 不缓存（每次访问都新建 session 拉取，适合开发调试）
    #   正数  —— 缓存过期秒数，过期后下一次访问自动 refresh
    # 负数会在 load_mcp_config / __post_init__ 阶段被拒绝（避免反直觉行为）。
    # MCP 规范要求 client 订阅 server 的 list_changed 通知做缓存失效，但当前
    # 架构是"每次新建短生命周期 session"，session 退出 cancel scope 也带走通知
    # 通道。TTL + `MCPClient.notify_list_changed` 应用层触发点共同兜住缓存陈旧
    # 问题。Long-lived listening session 是 follow-up（见 MCPClient docstring）。
    cache_ttl: float | None = 60.0

    def __post_init__(self) -> None:
        if self.cache_ttl is not None and self.cache_ttl < 0:
            raise ValueError(
                f"mcp config: server '{self.name}' cache_ttl must be >= 0 or "
                f"None, got {self.cache_ttl!r}"
            )


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
