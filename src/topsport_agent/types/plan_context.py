"""Plan 级共享状态容器。

类 LangGraph channel 模型的 typed 版本：用户继承 PlanContext 定义字段，
用 Annotated[T, Reducer(fn)] 在字段上声明合并语义；无 Reducer 的字段默认覆盖。

设计要点：
- 用 pydantic BaseModel 拿类型校验、JSON 序列化、IDE 支持
- reducer 通过 Annotated metadata 附着到字段，避免类体里维护第二份字典
- merge 不破坏不可变性：用 model_copy(update=...) 返回新实例，orchestrator 负责装回
- 序列化: model_dump_json() → Checkpointer 存盘；model_validate_json() → 恢复
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

__all__ = ["PlanContext", "Reducer"]


class Reducer:
    """字段级合并函数标记。

    用法：
        class MyCtx(PlanContext):
            notes: Annotated[list[str], Reducer(operator.add)] = []
            counter: Annotated[int, Reducer(lambda a, b: a + b)] = 0
            mode: str = "exploring"   # 无 Reducer → merge 时直接覆盖

    fn 接收 (current_value, new_value)，返回合并后的新值。
    fn 必须是纯函数：相同输入永远产出相同输出，不允许副作用，因为
    checkpoint 恢复 + 重放场景下同一次 merge 可能被执行多次。
    """

    __slots__ = ("fn",)

    def __init__(self, fn: Callable[[Any, Any], Any]) -> None:
        self.fn = fn

    def __repr__(self) -> str:
        return f"Reducer({getattr(self.fn, '__name__', repr(self.fn))})"


class PlanContext(BaseModel):
    """Plan 级共享状态基类。用户继承它声明自己的字段。

    不包含 plan-id / step-id 等编排元数据——那些由 Plan/PlanStep 承载。
    这里只放"任务相关"的共享数据，比如累积的搜索结果、评审轮次、flag。
    """

    model_config = ConfigDict(
        # 允许 merge 后 validate_assignment 重新跑字段校验
        validate_assignment=True,
        # 冻结禁用：merge 需要改字段；不可变语义由调用方用 model_copy 自行保证
        frozen=False,
        # 拒绝多余字段：PlanContext schema 是合同，打错字应该立刻暴露
        extra="forbid",
    )

    # ------------------------------------------------------------------
    # Reducer lookup
    # ------------------------------------------------------------------

    @classmethod
    def _find_reducer(cls, key: str) -> Reducer | None:
        """从字段 metadata 中找到绑定的 Reducer。

        语义：
        - 信任 merge() 已校验 key 存在（单点校验）
        - isinstance 严格过滤，避免与 pydantic Field / 第三方 Annotated metadata 碰撞
        - 多个 Reducer 抛错而非静默取首个，强制使用者把合并语义显式写在一个函数里
        """
        field_info = cls.model_fields[key]
        reducers = [m for m in field_info.metadata if isinstance(m, Reducer)]
        if len(reducers) > 1:
            raise ValueError(
                f"Field '{key}' declares {len(reducers)} reducers; only one is allowed"
            )
        return reducers[0] if reducers else None

    # ------------------------------------------------------------------
    # Merge API
    # ------------------------------------------------------------------

    def merge(self, key: str, value: Any) -> "PlanContext":
        """把 value 合并到 key 字段，返回新的 PlanContext 实例。

        - 若字段声明了 Reducer(fn)：new = fn(current, value)
        - 否则：new = value（覆盖）

        不可变语义：原实例不变，返回新实例。orchestrator 负责把新实例装回 Plan。
        """
        if key not in type(self).model_fields:
            raise KeyError(
                f"{type(self).__name__} has no field '{key}'. "
                f"Declared fields: {sorted(type(self).model_fields.keys())}"
            )
        reducer = type(self)._find_reducer(key)
        current = getattr(self, key)
        new_value = reducer.fn(current, value) if reducer else value
        # model_copy(update=...) 不走 validator（pydantic 设计选择）。
        # 改用 model_copy() + setattr 触发 __setattr__ → validate_assignment，
        # 确保 Field(ge=..., le=...) 等约束在 merge 路径也生效。
        copy = self.model_copy()
        setattr(copy, key, new_value)
        return copy

    def merge_many(self, updates: dict[str, Any]) -> "PlanContext":
        """批量合并。每个 key 独立走 merge；顺序按 dict 遍历序。"""
        ctx: PlanContext = self
        for key, value in updates.items():
            ctx = ctx.merge(key, value)
        return ctx

    # ------------------------------------------------------------------
    # Serialization helpers (Checkpointer 会用)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """返回 JSON 可序列化的字典，用于 Checkpointer 落盘。"""
        return self.model_dump(mode="json")

    @classmethod
    def restore(cls, data: dict[str, Any]) -> "PlanContext":
        """从 snapshot() 的输出恢复。失败会抛 pydantic ValidationError。"""
        return cls.model_validate(data)


# 泛型别名，供 Plan[CtxT] 使用
CtxT = TypeVar("CtxT", bound=PlanContext)


