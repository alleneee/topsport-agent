"""Orchestrator Phase 2b 集成测试：condition 过滤、post_condition 回跳、max_iterations 兜底。"""

from __future__ import annotations

import operator
from typing import Annotated

from topsport_agent.engine.orchestrator import Orchestrator, SubAgentConfig
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.plan import Plan, PlanStep, StepDecision, StepStatus
from topsport_agent.types.plan_context import PlanContext, Reducer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class ScriptedProvider:
    name = "scripted"

    def __init__(self, text: str = "done") -> None:
        self.text = text
        self.call_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request  # 不关心请求内容，纯脚本化应答
        self.call_count += 1
        return LLMResponse(text=self.text, finish_reason="stop")


class AlwaysAbort:
    """测试辅助：任何失败直接 abort，避免 orchestrator 阻塞等 provide_decision。"""

    name = "always-abort"

    async def handle_failure(self, plan, failed_steps):
        del plan, failed_steps
        return StepDecision.ABORT


class DemoCtx(PlanContext):
    mode: str = "exploring"
    attempts: Annotated[int, Reducer(operator.add)] = 0
    gate_open: bool = True


def _step(
    sid: str,
    deps: list[str] | None = None,
    *,
    condition=None,
    post_condition=None,
    max_iterations: int = 1,
) -> PlanStep:
    return PlanStep(
        id=sid,
        title=f"Step {sid}",
        instructions=f"do {sid}",
        depends_on=deps or [],
        condition=condition,
        post_condition=post_condition,
        max_iterations=max_iterations,
    )


async def _collect(agen) -> list[Event]:
    return [event async for event in agen]


def _config() -> SubAgentConfig:
    return SubAgentConfig(provider=ScriptedProvider(), model="fake")


def _types_of(events: list[Event]) -> list[EventType]:
    return [e.type for e in events]


# ---------------------------------------------------------------------------
# condition 过滤
# ---------------------------------------------------------------------------


async def test_condition_false_skips_step() -> None:
    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a", condition=lambda ctx: ctx.gate_open is False)],
        context=DemoCtx(gate_open=True),  # condition 返回 False
    )
    orch = Orchestrator(plan, _config())
    events = await _collect(orch.execute())

    assert EventType.PLAN_STEP_SKIPPED in _types_of(events)
    assert EventType.PLAN_STEP_START not in _types_of(events)
    assert plan.step_by_id("a").status == StepStatus.SKIPPED  # type: ignore[union-attr]
    assert plan.is_complete()


async def test_condition_true_runs_step() -> None:
    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a", condition=lambda ctx: ctx.gate_open is True)],
        context=DemoCtx(gate_open=True),
    )
    orch = Orchestrator(plan, _config())
    events = await _collect(orch.execute())

    assert EventType.PLAN_STEP_SKIPPED not in _types_of(events)
    assert EventType.PLAN_STEP_START in _types_of(events)
    assert plan.step_by_id("a").status == StepStatus.DONE  # type: ignore[union-attr]


async def test_condition_error_is_skipped_not_crashed() -> None:
    def _broken(ctx: DemoCtx) -> bool:
        raise RuntimeError("boom")

    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a", condition=_broken)],
        context=DemoCtx(),
    )
    orch = Orchestrator(plan, _config())
    events = await _collect(orch.execute())

    a = plan.step_by_id("a")
    assert a is not None
    assert a.status == StepStatus.SKIPPED
    assert a.error is not None and "boom" in a.error
    skipped_evs = [e for e in events if e.type == EventType.PLAN_STEP_SKIPPED]
    assert skipped_evs[0].payload["reason"] == "condition_error"


async def test_no_context_disables_condition_evaluation() -> None:
    """未传 context 时 condition 不求值（向后兼容）：即使写了 condition 也正常跑。"""
    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a", condition=lambda ctx: False)],   # 没 context，不会被求值
        context=None,
    )
    orch = Orchestrator(plan, _config())
    await _collect(orch.execute())
    assert plan.step_by_id("a").status == StepStatus.DONE  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# post_condition 回跳 + max_iterations
# ---------------------------------------------------------------------------


async def test_post_condition_true_ends_normally() -> None:
    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a", post_condition=lambda ctx: True)],
        context=DemoCtx(),
    )
    orch = Orchestrator(plan, _config())
    events = await _collect(orch.execute())

    assert EventType.PLAN_STEP_LOOP not in _types_of(events)
    assert plan.step_by_id("a").status == StepStatus.DONE  # type: ignore[union-attr]
    assert plan.step_by_id("a").iterations == 1  # type: ignore[union-attr]


async def test_post_condition_false_loops_until_max_then_fails() -> None:
    """post_condition 永假 + max_iterations=3 → 跑 3 次后 FAILED。"""
    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a", post_condition=lambda ctx: False, max_iterations=3)],
        context=DemoCtx(),
    )
    orch = Orchestrator(plan, _config(), failure_handlers=[AlwaysAbort()])
    events = await _collect(orch.execute())

    a = plan.step_by_id("a")
    assert a is not None
    assert a.iterations == 3
    assert a.status == StepStatus.FAILED
    assert a.error is not None and "post_condition not satisfied" in a.error
    # 应该有 2 次 LOOP 事件（第 1、第 2 次失败），第 3 次达到上限转为 FAILED
    loop_evs = [e for e in events if e.type == EventType.PLAN_STEP_LOOP]
    assert len(loop_evs) == 2
    # AlwaysAbort → PLAN_FAILED
    assert EventType.PLAN_FAILED in _types_of(events)


async def test_post_condition_flips_true_on_second_iteration() -> None:
    """第一次不满足，触发回跳；第二次满足，正常结束。"""
    # condition 读 ctx.attempts；外部用闭包模拟 sub-agent 的副作用：每次跑完后 attempts+=1
    calls = {"n": 0}

    def _post(ctx: DemoCtx) -> bool:
        calls["n"] += 1
        # 首次调用（n==1）返回 False，后续返回 True
        return calls["n"] >= 2

    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a", post_condition=_post, max_iterations=3)],
        context=DemoCtx(),
    )
    orch = Orchestrator(plan, _config())
    events = await _collect(orch.execute())

    a = plan.step_by_id("a")
    assert a is not None
    assert a.status == StepStatus.DONE
    assert a.iterations == 2
    loop_evs = [e for e in events if e.type == EventType.PLAN_STEP_LOOP]
    assert len(loop_evs) == 1


async def test_post_condition_loop_declares_downstream_in_reset_payload() -> None:
    """a 有 post_condition 回跳；b 依赖 a。
    a 第一次回跳时 b 还没 ready（依赖 a），但 reset_dependents_of 会把 b 纳入
    重置列表（PENDING 也会被"幂等重置"并出现在返回值）——这是为了让 orchestrator
    的订阅者能看到"这一轮影响了哪些下游"，便于 tracing 和调试。

    注意：此拓扑下 b 只会跑 1 次，因为 b 依赖 a，a 的每次回跳都发生在 b ready 之前。
    想验证"下游被重置重跑"需要一个 b 已经 DONE 又被回跳的场景，见下一条测试。
    """
    calls = {"n": 0}

    def _post(ctx: DemoCtx) -> bool:
        del ctx
        calls["n"] += 1
        return calls["n"] >= 2

    plan = Plan(
        id="p",
        goal="g",
        steps=[
            _step("a", post_condition=_post, max_iterations=3),
            _step("b", deps=["a"]),
        ],
        context=DemoCtx(),
    )
    orch = Orchestrator(plan, _config())
    events = await _collect(orch.execute())

    a = plan.step_by_id("a")
    b = plan.step_by_id("b")
    assert a is not None and b is not None
    assert a.status == StepStatus.DONE
    assert b.status == StepStatus.DONE
    assert a.iterations == 2
    assert b.iterations == 1   # b 只跑一次（此拓扑下 a 每次回跳发生在 b ready 之前）
    # LOOP 事件应声明 b 被纳入重置列表
    loop_evs = [e for e in events if e.type == EventType.PLAN_STEP_LOOP]
    assert len(loop_evs) == 1
    assert "b" in loop_evs[0].payload["reset_dependents"]


async def test_non_downstream_siblings_not_affected_by_upstream_loop() -> None:
    """和上游同波次但**不依赖**上游的 sibling 不会因为上游回跳而被重置。

    拓扑：a (post_condition 回跳), b (无依赖，与 a 同波次), c (depends_on=['a'])

    验证：a 回跳只影响 a 的反向可达集 {c}，b 独立完成一次即终态。
    这是 reset_dependents_of 走"反向 depends_on BFS"的语义守卫。
    """
    calls = {"n": 0}

    def _post_a(ctx: DemoCtx) -> bool:
        del ctx
        calls["n"] += 1
        return calls["n"] >= 3

    plan = Plan(
        id="p",
        goal="g",
        steps=[
            _step("a", post_condition=_post_a, max_iterations=5),
            _step("b"),                 # 与 a 同波次，不依赖 a
            _step("c", deps=["a"]),
        ],
        context=DemoCtx(),
    )
    orch = Orchestrator(plan, _config())
    events = await _collect(orch.execute())

    a = plan.step_by_id("a")
    b = plan.step_by_id("b")
    c = plan.step_by_id("c")
    assert a is not None and b is not None and c is not None
    assert a.iterations == 3
    assert b.iterations == 1    # b 不是 a 的下游，不受回跳影响
    assert c.iterations == 1    # c 依赖 a，a 最终 DONE 后才跑一次
    assert all(s.status == StepStatus.DONE for s in (a, b, c))

    # 每次 LOOP 事件的 reset_dependents 都应该包含 c（c 是 a 的下游），不应包含 b
    loop_evs = [e for e in events if e.type == EventType.PLAN_STEP_LOOP]
    assert len(loop_evs) == 2
    for ev in loop_evs:
        assert "c" in ev.payload["reset_dependents"]
        assert "b" not in ev.payload["reset_dependents"]


# ---------------------------------------------------------------------------
# 向后兼容
# ---------------------------------------------------------------------------


async def test_plain_plan_without_phase2b_fields_still_works() -> None:
    """不设 condition/post_condition/context —— 行为与 Phase 2a 之前完全一致。"""
    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a"), _step("b", deps=["a"])],
    )
    orch = Orchestrator(plan, _config())
    events = await _collect(orch.execute())

    assert EventType.PLAN_STEP_SKIPPED not in _types_of(events)
    assert EventType.PLAN_STEP_LOOP not in _types_of(events)
    assert plan.step_by_id("a").status == StepStatus.DONE  # type: ignore[union-attr]
    assert plan.step_by_id("b").status == StepStatus.DONE  # type: ignore[union-attr]
    assert plan.step_by_id("a").iterations == 1  # type: ignore[union-attr]
