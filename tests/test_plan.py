"""Tests for Plan data model and DAG validation."""

from __future__ import annotations

import pytest

from topsport_agent.types.plan import Plan, PlanStep, StepStatus


def _step(id: str, deps: list[str] | None = None) -> PlanStep:
    return PlanStep(
        id=id,
        title=f"Step {id}",
        instructions=f"Do {id}",
        depends_on=deps or [],
    )


class TestPlanValidation:
    def test_valid_linear_plan(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[_step("a"), _step("b", ["a"]), _step("c", ["b"])],
        )
        assert len(plan.steps) == 3

    def test_valid_diamond_dag(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[
                _step("a"),
                _step("b", ["a"]),
                _step("c", ["a"]),
                _step("d", ["b", "c"]),
            ],
        )
        assert len(plan.steps) == 4

    def test_valid_parallel_no_deps(self):
        plan = Plan(id="p1", goal="test", steps=[_step("a"), _step("b"), _step("c")])
        assert len(plan.steps) == 3

    def test_empty_plan_is_valid(self):
        plan = Plan(id="p1", goal="test", steps=[])
        assert plan.is_complete()

    def test_cycle_detected(self):
        with pytest.raises(ValueError, match="cycle"):
            Plan(
                id="p1",
                goal="test",
                steps=[_step("a", ["b"]), _step("b", ["a"])],
            )

    def test_self_dependency_detected(self):
        with pytest.raises(ValueError, match="depends on itself"):
            Plan(id="p1", goal="test", steps=[_step("a", ["a"])])

    def test_unknown_dependency_detected(self):
        with pytest.raises(ValueError, match="unknown step"):
            Plan(id="p1", goal="test", steps=[_step("a", ["x"])])

    def test_three_node_cycle(self):
        with pytest.raises(ValueError, match="cycle"):
            Plan(
                id="p1",
                goal="test",
                steps=[
                    _step("a", ["c"]),
                    _step("b", ["a"]),
                    _step("c", ["b"]),
                ],
            )


class TestReadySteps:
    def test_no_deps_all_ready(self):
        plan = Plan(id="p1", goal="test", steps=[_step("a"), _step("b")])
        ready = plan.ready_steps()
        assert {s.id for s in ready} == {"a", "b"}

    def test_deps_block_readiness(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[_step("a"), _step("b", ["a"])],
        )
        ready = plan.ready_steps()
        assert [s.id for s in ready] == ["a"]

    def test_completed_dep_unblocks(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[_step("a"), _step("b", ["a"])],
        )
        plan.steps[0].status = StepStatus.DONE
        ready = plan.ready_steps()
        assert [s.id for s in ready] == ["b"]

    def test_running_step_not_ready(self):
        plan = Plan(id="p1", goal="test", steps=[_step("a")])
        plan.steps[0].status = StepStatus.RUNNING
        assert plan.ready_steps() == []

    def test_diamond_parallel_middle(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[
                _step("a"),
                _step("b", ["a"]),
                _step("c", ["a"]),
                _step("d", ["b", "c"]),
            ],
        )
        plan.steps[0].status = StepStatus.DONE
        ready = plan.ready_steps()
        assert {s.id for s in ready} == {"b", "c"}


class TestCompletionAndFailure:
    def test_is_complete_when_all_done(self):
        plan = Plan(id="p1", goal="test", steps=[_step("a"), _step("b")])
        plan.steps[0].status = StepStatus.DONE
        plan.steps[1].status = StepStatus.DONE
        assert plan.is_complete()

    def test_is_complete_with_skipped(self):
        plan = Plan(id="p1", goal="test", steps=[_step("a"), _step("b")])
        plan.steps[0].status = StepStatus.DONE
        plan.steps[1].status = StepStatus.SKIPPED
        assert plan.is_complete()

    def test_not_complete_with_pending(self):
        plan = Plan(id="p1", goal="test", steps=[_step("a"), _step("b")])
        plan.steps[0].status = StepStatus.DONE
        assert not plan.is_complete()

    def test_has_failed(self):
        plan = Plan(id="p1", goal="test", steps=[_step("a")])
        plan.steps[0].status = StepStatus.FAILED
        assert plan.has_failed()

    def test_step_by_id(self):
        plan = Plan(id="p1", goal="test", steps=[_step("a"), _step("b")])
        assert plan.step_by_id("b") is plan.steps[1]
        assert plan.step_by_id("x") is None


class TestSkipDependents:
    def test_skip_direct_dependent(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[_step("a"), _step("b", ["a"])],
        )
        plan.steps[0].status = StepStatus.FAILED
        skipped = plan.skip_dependents_of("a")
        assert skipped == ["b"]
        assert plan.steps[1].status == StepStatus.SKIPPED

    def test_skip_transitive_dependents(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[_step("a"), _step("b", ["a"]), _step("c", ["b"])],
        )
        plan.steps[0].status = StepStatus.FAILED
        skipped = plan.skip_dependents_of("a")
        assert skipped == ["b", "c"]

    def test_skip_does_not_affect_done_steps(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[
                _step("a"),
                _step("b", ["a"]),
                _step("c", ["a"]),
            ],
        )
        plan.steps[0].status = StepStatus.FAILED
        plan.steps[1].status = StepStatus.DONE
        skipped = plan.skip_dependents_of("a")
        assert skipped == ["c"]
        assert plan.steps[1].status == StepStatus.DONE

    def test_skip_partial_dag(self):
        plan = Plan(
            id="p1",
            goal="test",
            steps=[
                _step("a"),
                _step("b"),
                _step("c", ["a"]),
                _step("d", ["b"]),
            ],
        )
        plan.steps[0].status = StepStatus.FAILED
        skipped = plan.skip_dependents_of("a")
        assert skipped == ["c"]
        assert plan.steps[3].status == StepStatus.PENDING
