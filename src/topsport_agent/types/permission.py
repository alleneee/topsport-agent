"""Tool permission 决策模型。对标 claude-code 的 PermissionBehavior/Result。

核心模型：
- PermissionBehavior：allow / deny / ask
- PermissionDecision：behavior + 可选 reason + 可选 updated_input（可改写 LLM 传的参数）
- PermissionChecker：Engine 在 handler 前询问"本次调用是否允许"的 Protocol
- PermissionAsker：checker 返回 ask 时由 Engine 调用，让外部（CLI 终端、server
  SSE 提示、pentest-style 自动化脚本）决定最终 allow/deny

最小可用版（本轮只做）：
1. 全部是运行时注入，Engine 不感知具体实现
2. 默认 checker 基于 ToolSpec 字段：destructive=True → ask；其它 → allow
3. 不支持规则持久化、不读 settings.json、不跨 session 记住选择——这些留给调用方
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .message import ToolCall
    from .tool import ToolContext, ToolSpec

__all__ = [
    "PermissionAsker",
    "PermissionBehavior",
    "PermissionChecker",
    "PermissionDecision",
    "allow",
    "ask",
    "deny",
]


class PermissionBehavior(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(slots=True, frozen=True)
class PermissionDecision:
    """Permission 决策结果。

    - `behavior`: ALLOW / DENY / ASK
    - `reason`: DENY/ASK 时给 LLM 看的解释，也可作为 asker 的上下文
    - `updated_input`: ALLOW 时可非空——checker 可以改写 LLM 原参数
      （例：把相对路径展开为绝对、注入额外安全标志）。None 表示沿用原参数。
    """

    behavior: PermissionBehavior
    reason: str | None = None
    updated_input: dict[str, Any] | None = None


# 三个 helper 让调用代码读起来像 DSL：`return allow()` / `return deny("...")`
def allow(updated_input: dict[str, Any] | None = None) -> PermissionDecision:
    return PermissionDecision(PermissionBehavior.ALLOW, updated_input=updated_input)


def deny(reason: str) -> PermissionDecision:
    return PermissionDecision(PermissionBehavior.DENY, reason=reason)


def ask(reason: str | None = None) -> PermissionDecision:
    return PermissionDecision(PermissionBehavior.ASK, reason=reason)


class PermissionChecker(Protocol):
    """同步决策接口：Engine 在工具 handler 前调用。

    返回 PermissionBehavior.ASK 时 Engine 会转给 PermissionAsker；没有 asker
    时默认按 DENY 处理（保守——宁可错杀不可放行 destructive 工具）。
    """

    name: str

    async def check(
        self,
        tool: "ToolSpec",
        call: "ToolCall",
        context: "ToolContext",
    ) -> PermissionDecision: ...


class PermissionAsker(Protocol):
    """交互式最终裁决接口。CLI 实现读 stdin，server 实现走 SSE prompt。

    必须返回 ALLOW 或 DENY；asker 再返回 ASK 是逻辑错误，Engine 当 DENY 处理。
    """

    name: str

    async def ask(
        self,
        tool: "ToolSpec",
        call: "ToolCall",
        context: "ToolContext",
        reason: str | None,
    ) -> PermissionDecision: ...


# 便捷类型别名：允许直接传 callable（不强制类化）
PermissionCheckFn = Callable[
    ["ToolSpec", "ToolCall", "ToolContext"], Awaitable[PermissionDecision]
]
