"""默认 permission checker 实现。

基于 ToolSpec 元数据（destructive / read_only / trust_level）做开箱即用的策略：
- destructive=True → ASK（必须经 asker 二次确认）
- read_only=True → ALLOW（纯查询总是放行）
- 其它 → ALLOW（向后兼容，不阻塞现有流程）

生产接入复杂策略（基于 tenant、基于参数正则、基于历史决策缓存）时，自己实现
PermissionChecker Protocol 替换即可；不要在这里加业务逻辑。
"""

from __future__ import annotations

from ...types.permission import (
    PermissionAsker,
    PermissionDecision,
    allow,
    ask,
)
from ...types.message import ToolCall
from ...types.tool import ToolContext, ToolSpec

__all__ = ["AlwaysAskAsker", "AlwaysDenyAsker", "DefaultPermissionChecker"]


class DefaultPermissionChecker:
    """最小可用策略。"""

    name = "default"

    async def check(
        self,
        tool: ToolSpec,
        call: ToolCall,
        context: ToolContext,
    ) -> PermissionDecision:
        del call, context
        if getattr(tool, "destructive", False):
            return ask(f"tool '{tool.name}' is marked destructive")
        return allow()


class AlwaysDenyAsker:
    """保守默认 asker：一切 ASK 都 DENY。适合非交互环境（batch、CI）。"""

    name = "always-deny"

    async def ask(
        self,
        tool: ToolSpec,
        call: ToolCall,
        context: ToolContext,
        reason: str | None,
    ) -> PermissionDecision:
        del call, context
        from ...types.permission import deny

        return deny(reason or f"destructive tool '{tool.name}' denied by default")


class AlwaysAskAsker:
    """测试/dev 环境：一切 ASK 都 ALLOW。生产不要用。"""

    name = "always-allow"

    async def ask(
        self,
        tool: ToolSpec,
        call: ToolCall,
        context: ToolContext,
        reason: str | None,
    ) -> PermissionDecision:
        del tool, call, context, reason
        return allow()


# ---------------------------------------------------------------------------
# Asker Protocol self-check
# ---------------------------------------------------------------------------

_ASKERS: tuple[PermissionAsker, ...] = (AlwaysDenyAsker(), AlwaysAskAsker())
assert all(hasattr(a, "ask") and hasattr(a, "name") for a in _ASKERS)
