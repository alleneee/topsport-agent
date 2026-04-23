"""Plan.reset_dependents_of 契约测试（Phase 2b 循环语义）。

覆盖 TODO 里列出的每一条边界决策，等 reset_dependents_of 实现后变绿。
"""

from __future__ import annotations

import pytest

from topsport_agent.types.plan import Plan, PlanStep, StepStatus


def _plan_linear() -> Plan:
    """a -> b -> c 线性链；便于测试传递下游。"""
    return Plan(
        id="p",
        goal="g",
        steps=[
            PlanStep(id="a", title="A", instructions=""),
            PlanStep(id="b", title="B", instructions="", depends_on=["a"]),
            PlanStep(id="c", title="C", instructions="", depends_on=["b"]),
        ],
    )


def _plan_fork() -> Plan:
    """a -> {b1, b2} -> c；b1 和 b2 都依赖 a，c 依赖两者。"""
    return Plan(
        id="p",
        goal="g",
        steps=[
            PlanStep(id="a", title="A", instructions=""),
            PlanStep(id="b1", title="B1", instructions="", depends_on=["a"]),
            PlanStep(id="b2", title="B2", instructions="", depends_on=["a"]),
            PlanStep(id="c", title="C", instructions="", depends_on=["b1", "b2"]),
        ],
    )


# ---------------------------------------------------------------------------
# 范围：传递下游（BFS）
# ---------------------------------------------------------------------------


def test_reset_propagates_transitively() -> None:
    """a 被触发重置 → b（直接下游）和 c（传递下游）都要被重置。"""
    plan = _plan_linear()
    for s in plan.steps:
        s.status = StepStatus.DONE
        s.result = f"result-{s.id}"
    reset_ids = plan.reset_dependents_of("a")
    assert set(reset_ids) == {"b", "c"}
    assert plan.step_by_id("b").status == StepStatus.PENDING  # type: ignore[union-attr]
    assert plan.step_by_id("c").status == StepStatus.PENDING  # type: ignore[union-attr]


def test_reset_covers_all_branches_of_fork() -> None:
    """fork 拓扑下，a 重置 → b1/b2/c 全部重置。"""
    plan = _plan_fork()
    for s in plan.steps:
        s.status = StepStatus.DONE
    reset_ids = plan.reset_dependents_of("a")
    assert set(reset_ids) == {"b1", "b2", "c"}


def test_reset_does_not_touch_source_step() -> None:
    """reset_dependents_of('a') 只动下游，a 本身由 orchestrator 管理。"""
    plan = _plan_linear()
    for s in plan.steps:
        s.status = StepStatus.DONE
    plan.reset_dependents_of("a")
    assert plan.step_by_id("a").status == StepStatus.DONE  # type: ignore[union-attr]


def test_reset_does_not_touch_unrelated_steps() -> None:
    """不是下游的 step 不受影响。"""
    plan = Plan(
        id="p",
        goal="g",
        steps=[
            PlanStep(id="a", title="A", instructions=""),
            PlanStep(id="b", title="B", instructions="", depends_on=["a"]),
            PlanStep(id="x", title="X", instructions=""),   # 独立节点
        ],
    )
    for s in plan.steps:
        s.status = StepStatus.DONE
    plan.reset_dependents_of("a")
    assert plan.step_by_id("x").status == StepStatus.DONE  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 状态机：哪些状态可被重置 / 哪些豁免
# ---------------------------------------------------------------------------


def test_running_dependent_is_not_reset() -> None:
    """RUNNING 的下游正在跑，不能中途改状态；orchestrator 下一轮会重新评估。"""
    plan = _plan_linear()
    plan.step_by_id("a").status = StepStatus.DONE  # type: ignore[union-attr]
    plan.step_by_id("b").status = StepStatus.RUNNING  # type: ignore[union-attr]
    plan.step_by_id("c").status = StepStatus.PENDING  # type: ignore[union-attr]

    reset_ids = plan.reset_dependents_of("a")
    assert "b" not in reset_ids
    assert plan.step_by_id("b").status == StepStatus.RUNNING  # type: ignore[union-attr]


def test_skipped_dependent_is_not_revived() -> None:
    """SKIPPED（condition 已判过）不该被自动复活。"""
    plan = _plan_linear()
    plan.step_by_id("a").status = StepStatus.DONE  # type: ignore[union-attr]
    plan.step_by_id("b").status = StepStatus.SKIPPED  # type: ignore[union-attr]

    reset_ids = plan.reset_dependents_of("a")
    assert "b" not in reset_ids
    assert plan.step_by_id("b").status == StepStatus.SKIPPED  # type: ignore[union-attr]


def test_done_dependent_is_reset_and_result_cleared() -> None:
    """DONE 的下游消费了旧 result，必须重置；旧 result/error 清空。"""
    plan = _plan_linear()
    b = plan.step_by_id("b")
    assert b is not None
    plan.step_by_id("a").status = StepStatus.DONE  # type: ignore[union-attr]
    b.status = StepStatus.DONE
    b.result = "stale output"
    b.error = None

    plan.reset_dependents_of("a")
    assert b.status == StepStatus.PENDING
    assert b.result is None


# ---------------------------------------------------------------------------
# iterations 归零决策：保留（max_iterations 是硬上限）
# ---------------------------------------------------------------------------


def test_reset_preserves_iterations_counter() -> None:
    """iterations 不归零：max_iterations 是硬上限，避免循环失控。"""
    plan = _plan_linear()
    b = plan.step_by_id("b")
    assert b is not None
    plan.step_by_id("a").status = StepStatus.DONE  # type: ignore[union-attr]
    b.status = StepStatus.DONE
    b.iterations = 2   # b 之前已经跑过 2 次
    b.max_iterations = 5

    plan.reset_dependents_of("a")
    assert b.iterations == 2   # 不归零
    assert b.status == StepStatus.PENDING


# ---------------------------------------------------------------------------
# 未知 step_id 的鲁棒性
# ---------------------------------------------------------------------------


def test_reset_unknown_step_raises_or_noop() -> None:
    """未知 step_id 应抛 KeyError 或返回空——两者择一，不要静默通过。

    这里只要求"不要静默返回 ['a','b','c'] 当成根节点处理"。
    """
    plan = _plan_linear()
    for s in plan.steps:
        s.status = StepStatus.DONE

    try:
        reset_ids = plan.reset_dependents_of("no-such-step")
    except (KeyError, ValueError):
        # 允许抛错
        return
    # 或允许返回空列表
    assert reset_ids == [], (
        f"unknown step_id 应抛错或返回 []，实际返回 {reset_ids} "
        "— 不要把未知 id 当根节点重置全图"
    )


# ---------------------------------------------------------------------------
# PlanStep 新字段默认值
# ---------------------------------------------------------------------------


def test_plan_step_new_fields_have_safe_defaults() -> None:
    """condition/post_condition 默认 None，max_iterations=1，iterations=0。"""
    step = PlanStep(id="x", title="X", instructions="")
    assert step.condition is None
    assert step.post_condition is None
    assert step.max_iterations == 1
    assert step.iterations == 0


def test_plan_accepts_optional_context() -> None:
    """Plan.context 可选；未传时为 None，旧行为不变。"""
    from topsport_agent.types.plan_context import PlanContext

    class MyCtx(PlanContext):
        counter: int = 0

    plan_no_ctx = Plan(id="p", goal="g", steps=[PlanStep(id="a", title="A", instructions="")])
    assert plan_no_ctx.context is None

    plan_with_ctx = Plan(
        id="p",
        goal="g",
        steps=[PlanStep(id="a", title="A", instructions="")],
        context=MyCtx(counter=5),
    )
    assert plan_with_ctx.context is not None
    assert plan_with_ctx.context.counter == 5  # type: ignore[attr-defined]


# sanity: 现有 skip_dependents_of 不受影响
def test_skip_dependents_still_works() -> None:
    plan = _plan_linear()
    plan.step_by_id("a").status = StepStatus.FAILED  # type: ignore[union-attr]
    skipped = plan.skip_dependents_of("a")
    assert set(skipped) == {"b", "c"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
