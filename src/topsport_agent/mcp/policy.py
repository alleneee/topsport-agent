"""MCP stdio 启动安全策略：白名单 + shell 解释器拒绝。

默认 permissive（兼容历史 Claude Desktop 配置），加 warning 日志。
JSON 配置文件出现 `allowlist` 字段即自动切 strict，此时每个 stdio server
必须匹配一条 (name, command_abs, args_prefix) 条目。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_logger = logging.getLogger(__name__)

# basename 命中即拒绝（strict），或告警（permissive）。
# 这些解释器允许通过 `-c` / `-e` / 脚本参数执行任意代码，是 MCP stdio 的主要 RCE 入口。
SHELL_INTERPRETERS = frozenset(
    {
        "sh", "bash", "zsh", "ash", "dash", "ksh", "fish", "csh", "tcsh",
        "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
        "busybox",
    }
)


@dataclass(slots=True, frozen=True)
class AllowEntry:
    """MCP stdio 白名单一条记录。

    command 必须是绝对路径；args_prefix 为空元组代表对 args 无限制。
    运行时匹配规则：config.name == entry.name 且 config.command == entry.command
    且 config.args 以 entry.args_prefix 为前缀。
    """

    name: str
    command: str
    args_prefix: tuple[str, ...] = ()

    def matches(self, *, server_name: str, command: str, args: list[str]) -> bool:
        if self.name != server_name:
            return False
        if self.command != command:
            return False
        if len(args) < len(self.args_prefix):
            return False
        return tuple(args[: len(self.args_prefix)]) == self.args_prefix


@dataclass(slots=True, frozen=True)
class MCPSecurityPolicy:
    """两档：permissive 只告警不阻止；strict 强制白名单。"""

    mode: Literal["strict", "permissive"] = "permissive"
    allowlist: tuple[AllowEntry, ...] = field(default_factory=tuple)

    @classmethod
    def strict(cls, entries: list[AllowEntry]) -> "MCPSecurityPolicy":
        return cls(mode="strict", allowlist=tuple(entries))

    @classmethod
    def permissive(cls) -> "MCPSecurityPolicy":
        return cls(mode="permissive", allowlist=())


class MCPPolicyViolation(ValueError):
    """Strict 策略下任何违反规则的配置都抛出此异常，阻止进一步加载。"""


def enforce_stdio_policy(
    *,
    server_name: str,
    command: str | None,
    args: list[str],
    policy: MCPSecurityPolicy,
) -> None:
    """stdio 启动前的策略检查。

    - strict 模式：违规直接抛 MCPPolicyViolation
    - permissive 模式：只记 warning，便于未来升级为 strict 前观察流量

    合规的 HTTP 传输跳过此检查（由不同的 allowlist 机制处理，见 SEC-H-02）。
    """
    if not command:
        # config 阶段已校验 stdio 必有 command，这里兜底。
        raise MCPPolicyViolation(
            f"mcp policy: server {server_name!r} stdio requires 'command'"
        )

    basename = Path(command).name.lower()
    is_absolute = Path(command).is_absolute()

    violations: list[str] = []
    if basename in SHELL_INTERPRETERS:
        violations.append(
            f"command basename {basename!r} is a shell interpreter; "
            "allows arbitrary code execution via -c / script args"
        )
    if not is_absolute:
        violations.append(
            f"command {command!r} must be an absolute path "
            "(PATH lookup is not allowed under strict policy)"
        )

    if policy.mode == "strict":
        if not any(
            entry.matches(server_name=server_name, command=command, args=args)
            for entry in policy.allowlist
        ):
            violations.append(
                "no allowlist entry matches (name, command, args_prefix)"
            )
        if violations:
            raise MCPPolicyViolation(
                f"mcp policy [strict]: server {server_name!r} rejected: "
                + "; ".join(violations)
            )
        return

    # permissive: 仅告警，不阻止
    if violations:
        _logger.warning(
            "mcp policy [permissive]: server %r would be rejected under strict: %s",
            server_name,
            "; ".join(violations),
        )
