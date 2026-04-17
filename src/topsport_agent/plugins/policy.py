"""插件 hooks 执行安全策略：argv 强制 + 可信插件白名单。

两档模式：
- permissive（默认，兼容历史 `command: "..."` 字符串写法）：
  legacy 字符串 command 会被 shlex 分词执行，但记 warning 告知调用方迁移到 argv 列表。
  shell 特殊语法（pipes/重定向/命令替换）在 shlex 分词下失效 —— 这是期望行为，
  因为那些正是攻击面。

- strict：
  只接受 `command: ["argv", ...]` 列表形式。legacy 字符串 command 直接抛
  PluginPolicyViolation。且每个插件必须出现在 trusted_plugins 中才允许加载 hooks。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PluginSecurityPolicy:
    """插件 hooks 执行策略。"""

    mode: Literal["strict", "permissive"] = "permissive"
    trusted_plugins: frozenset[str] = frozenset()

    @classmethod
    def strict(cls, trusted: list[str]) -> "PluginSecurityPolicy":
        return cls(mode="strict", trusted_plugins=frozenset(trusted))

    @classmethod
    def permissive(cls) -> "PluginSecurityPolicy":
        return cls(mode="permissive", trusted_plugins=frozenset())


class PluginPolicyViolation(ValueError):
    """Strict 策略下任何违规命令或未授信插件都抛此异常，阻止 hook 注册。"""


def enforce_plugin_loadable(
    *,
    plugin_name: str,
    policy: PluginSecurityPolicy,
) -> bool:
    """插件是否允许加载 hooks。
    - strict：必须在 trusted_plugins 中
    - permissive：始终允许（不可信插件只记 warning）
    返回 True 允许，False 拒绝但不抛异常（调用方决定是否继续处理其它插件）。
    """
    if plugin_name in policy.trusted_plugins:
        return True
    if policy.mode == "strict":
        _logger.warning(
            "plugin policy [strict]: rejecting hooks from untrusted plugin %r",
            plugin_name,
        )
        return False
    _logger.warning(
        "plugin policy [permissive]: plugin %r hooks loaded without trust check; "
        "add to trusted_plugins before switching to strict",
        plugin_name,
    )
    return True


def enforce_command_shape(
    *,
    plugin_name: str,
    event_name: str,
    command: object,
    policy: PluginSecurityPolicy,
) -> list[str] | None:
    """检查 hook command 字段是否符合策略，返回规范化后的 argv 列表。

    - list[str] -> 直接返回（首选 / 强制的安全形式）
    - str       -> permissive: 返回 None 让调用方用 shlex.split + warning；
                   strict: 抛 PluginPolicyViolation
    - 其它      -> 始终抛异常
    """
    if isinstance(command, list) and all(isinstance(a, str) for a in command):
        return list(command)

    if isinstance(command, str):
        if policy.mode == "strict":
            raise PluginPolicyViolation(
                f"plugin policy [strict]: plugin {plugin_name!r} event "
                f"{event_name!r} uses legacy string command which allows "
                "shell interpretation; use argv list form: "
                f'"command": ["program", "arg1", "arg2"]'
            )
        # permissive: 告警，返回 None 让调用方继续做 shlex.split
        _logger.warning(
            "plugin policy [permissive]: plugin %r event %r uses legacy string "
            "command; shlex-splitting but will not support pipes / redirects / "
            "substitutions. Migrate to argv list.",
            plugin_name,
            event_name,
        )
        return None

    raise PluginPolicyViolation(
        f"plugin policy: plugin {plugin_name!r} event {event_name!r} "
        f"command must be a list[str] or str, got {type(command).__name__}"
    )
