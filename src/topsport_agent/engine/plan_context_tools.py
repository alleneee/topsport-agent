"""plan_context_read / plan_context_merge 工具：让 sub-agent 读写共享 PlanContext。

为何要这个桥：sub-agent 的 Engine 是独立的 ReAct 会话，看不到 Plan 层。但
condition / post_condition 的判定依赖 context，而 context 的写入来源只有 sub-agent
（它是唯一会"根据任务执行产生信息"的地方）。因此我们给 sub-agent 两个工具：
- `plan_context_read()`：查当前 context 快照，让 LLM 知道共享状态。
- `plan_context_merge(key, value)`：把结果按字段 reducer 合并进 context。

并发注意：同一波次里多个 step 并发跑（asyncio.gather），可能并发写 context。
Bridge 内部用 asyncio.Lock 串行化 merge，保证读-改-写原子，不会丢更新。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..types.plan import Plan
from ..types.tool import ToolContext, ToolSpec

__all__ = ["PlanContextBridge", "PlanContextToolSource"]

_logger = logging.getLogger(__name__)


class PlanContextBridge:
    """封装 Plan.context 读写的并发安全代理。

    生命周期：每个 Orchestrator 一个 bridge 实例，绑定到同一个 Plan 对象。
    Plan.context 可能被 merge 替换为新实例（merge 返回新 PlanContext）；
    bridge 通过 `self._plan.context` 动态读取，始终反映最新值。
    """

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self._lock = asyncio.Lock()

    async def read(self) -> dict[str, Any]:
        """返回当前 context snapshot；无 context 时返回空 dict（而非 None），方便 LLM 消费。"""
        ctx = self._plan.context
        if ctx is None:
            return {}
        return ctx.snapshot()

    async def merge(self, key: str, value: Any) -> dict[str, Any]:
        """按 reducer 合并 value 到 key 字段，返回合并后的完整 snapshot。

        - 无 context → 抛 ValueError（不静默创建，避免 LLM 误用）
        - 未知 key / 校验失败 → 让 PlanContext.merge 的原生异常冒出去，由 Engine
          的 tool handler 捕获，作为 tool_result.is_error 返回给 LLM——LLM 看到
          "field 'xxx' not declared" 就能自我纠正。
        """
        async with self._lock:
            current = self._plan.context
            if current is None:
                raise ValueError(
                    "plan has no PlanContext; pass `context=MyCtx()` when constructing "
                    "Plan to enable plan_context tools"
                )
            merged = current.merge(key, value)
            self._plan.context = merged
            _logger.debug("plan_context merge: plan=%s key=%s", self._plan.id, key)
            return merged.snapshot()

    # ------------------------------------------------------------------
    # ToolSpec factories
    # ------------------------------------------------------------------

    def make_read_tool(self) -> ToolSpec:
        async def _handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            del args, ctx
            return await self.read()

        return ToolSpec(
            name="plan_context_read",
            description=(
                "Read the current shared PlanContext for this run. "
                "Returns a JSON object with all declared context fields and their current values. "
                "Use this to check what the plan knows so far before deciding what to do next."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=_handler,
            trust_level="trusted",
        )

    def make_merge_tool(self) -> ToolSpec:
        async def _handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            del ctx
            try:
                key = args["key"]
                value = args["value"]
            except KeyError as exc:
                raise ValueError(
                    f"plan_context_merge requires 'key' and 'value' arguments; missing {exc}"
                ) from exc
            updated = await self.merge(key, value)
            return {"merged": True, "field": key, "context": updated}

        return ToolSpec(
            name="plan_context_merge",
            description=(
                "Merge a value into a specific PlanContext field using the field's "
                "declared reducer (e.g. list-append, integer-sum) or overwrite if no "
                "reducer is declared. Returns the full context snapshot after merging. "
                "Use this to publish your step's findings so downstream steps and "
                "post_condition checks can see them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The context field name to merge into.",
                    },
                    "value": {
                        "description": (
                            "The value to merge. Must match the field's declared type. "
                            "JSON-serializable: string, number, boolean, list, or object."
                        ),
                    },
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            handler=_handler,
            trust_level="trusted",
        )

    def make_tools(self) -> list[ToolSpec]:
        return [self.make_read_tool(), self.make_merge_tool()]


class PlanContextToolSource:
    """把 bridge 的两个工具作为 ToolSource 暴露，让 Engine._snapshot_tools 每步拿到。

    工具是稳定的（bridge 一致），`list_tools` 每次返回同样两个实例即可。
    """

    name = "plan_context"

    def __init__(self, bridge: PlanContextBridge) -> None:
        self._bridge = bridge
        # 工具实例在 bridge 里构造一次，复用；避免每步新建闭包产生 GC 压力。
        self._tools = bridge.make_tools()

    async def list_tools(self) -> list[ToolSpec]:
        return list(self._tools)


# ---------------------------------------------------------------------------
# Helper (for tests / scripts)
# ---------------------------------------------------------------------------


def ensure_json_serializable(value: Any) -> Any:
    """辅助：调用方把 value 喂给 merge 前可选地做一次 JSON round-trip，
    把不可序列化的内部对象（例如 pydantic 实例）降级为 dict。
    Engine handler 本身不强制调用；留作扩展点。
    """
    return json.loads(json.dumps(value, default=str))
