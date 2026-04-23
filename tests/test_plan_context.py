"""PlanContext + Reducer 单元测试。覆盖 merge 的两种语义、不可变、序列化、边界。"""

from __future__ import annotations

import operator
from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from topsport_agent.types.plan_context import PlanContext, Reducer


class DemoCtx(PlanContext):
    notes: Annotated[list[str], Reducer(operator.add)] = []
    counter: Annotated[int, Reducer(lambda a, b: a + b)] = 0
    mode: str = "exploring"
    tags: Annotated[set[str], Reducer(lambda a, b: a | b)] = set()


# ---------------------------------------------------------------------------
# Reducer 语义
# ---------------------------------------------------------------------------


def test_reducer_appends_list() -> None:
    ctx = DemoCtx()
    ctx2 = ctx.merge("notes", ["hello"])
    ctx3 = ctx2.merge("notes", ["world"])
    assert ctx3.notes == ["hello", "world"]


def test_reducer_sums_counter() -> None:
    ctx = DemoCtx(counter=10)
    ctx2 = ctx.merge("counter", 5)
    assert ctx2.counter == 15


def test_reducer_unions_set() -> None:
    ctx = DemoCtx(tags={"a"})
    ctx2 = ctx.merge("tags", {"b", "c"})
    assert ctx2.tags == {"a", "b", "c"}


def test_no_reducer_overrides() -> None:
    ctx = DemoCtx(mode="exploring")
    ctx2 = ctx.merge("mode", "refining")
    assert ctx2.mode == "refining"


# ---------------------------------------------------------------------------
# 不可变性
# ---------------------------------------------------------------------------


def test_merge_returns_new_instance() -> None:
    ctx = DemoCtx(notes=["seed"])
    ctx2 = ctx.merge("notes", ["delta"])
    assert ctx is not ctx2
    assert ctx.notes == ["seed"]
    assert ctx2.notes == ["seed", "delta"]


def test_merge_many_chains_in_order() -> None:
    ctx = DemoCtx()
    ctx2 = ctx.merge_many({"counter": 1, "mode": "a", "notes": ["x"]})
    assert ctx2.counter == 1
    assert ctx2.mode == "a"
    assert ctx2.notes == ["x"]


# ---------------------------------------------------------------------------
# 边界 / 错误
# ---------------------------------------------------------------------------


def test_merge_unknown_key_raises_key_error() -> None:
    ctx = DemoCtx()
    with pytest.raises(KeyError) as ei:
        ctx.merge("nonexistent", 1)
    # 错误消息应列出合法字段帮助调试
    assert "nonexistent" in str(ei.value)
    assert "notes" in str(ei.value)


def test_extra_field_on_construction_rejected() -> None:
    with pytest.raises(ValidationError):
        DemoCtx(unknown_field=42)  # type: ignore[call-arg]


def test_type_coercion_still_validates() -> None:
    ctx = DemoCtx()
    # counter 是 int；注入非数值 reducer 会抛（lambda 里 + 运算对 str 无意义）
    with pytest.raises(TypeError):
        ctx.merge("counter", "not-a-number")


# ---------------------------------------------------------------------------
# 序列化 / 恢复
# ---------------------------------------------------------------------------


def test_snapshot_roundtrip() -> None:
    ctx = DemoCtx(notes=["a", "b"], counter=3, mode="done", tags={"x"})
    data = ctx.snapshot()
    restored = DemoCtx.restore(data)
    assert restored == ctx


def test_snapshot_is_json_serializable() -> None:
    import json

    ctx = DemoCtx(notes=["a"], counter=1, tags={"x", "y"})
    # mode="json" 保证 set → list，JSON 可编码
    json_str = json.dumps(ctx.snapshot())
    restored = DemoCtx.restore(json.loads(json_str))
    assert restored.notes == ctx.notes
    assert restored.counter == ctx.counter
    assert restored.tags == ctx.tags


# ---------------------------------------------------------------------------
# 决定 2/3：多 Reducer 抛错 + Field 混排不误伤
# ---------------------------------------------------------------------------


def test_multiple_reducers_on_same_field_raise() -> None:
    class BadCtx(PlanContext):
        score: Annotated[int, Reducer(max), Reducer(operator.add)] = 0

    ctx = BadCtx()
    with pytest.raises(ValueError) as ei:
        ctx.merge("score", 5)
    # 报错消息要能定位具体字段，方便调试
    assert "score" in str(ei.value)
    assert "2 reducers" in str(ei.value)


def test_field_constraint_does_not_confuse_reducer_lookup() -> None:
    """pydantic Field 约束与 Reducer 同在 metadata 时，只有 Reducer 被识别。"""

    class MixedCtx(PlanContext):
        counter: Annotated[
            int,
            Field(ge=0, le=100, description="bounded counter"),
            Reducer(operator.add),
        ] = 0
        bounded: Annotated[int, Field(ge=0)] = 0  # 有 Field，无 Reducer → 覆盖语义

    ctx = MixedCtx(counter=5, bounded=3)
    ctx2 = ctx.merge("counter", 10)   # reducer 生效：5 + 10 = 15
    assert ctx2.counter == 15

    ctx3 = ctx2.merge("bounded", 7)   # 无 reducer：覆盖
    assert ctx3.bounded == 7

    # Field 约束在 merge 后仍生效（validate_assignment=True）
    with pytest.raises(ValidationError):
        ctx3.merge("bounded", -1)
