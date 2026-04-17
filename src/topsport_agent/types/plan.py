from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepDecision(StrEnum):
    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"


@dataclass(slots=True)
class PlanStep:
    id: str
    title: str
    instructions: str
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None


@dataclass(slots=True)
class Plan:
    id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        """构造即校验：不合法的 DAG 不允许存在，避免下游调度器拿到带环或断链的计划。"""
        self.validate()

    def validate(self) -> None:
        ids = {s.id for s in self.steps}
        for step in self.steps:
            if step.id in step.depends_on:
                raise ValueError(f"Step '{step.id}' depends on itself")
            for dep in step.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on unknown step '{dep}'"
                    )
        self._topological_order()

    def _topological_order(self) -> list[str]:
        """Kahn 拓扑排序：结果长度 < 步骤数说明存在环，直接拒绝。"""
        in_degree: dict[str, int] = {s.id: 0 for s in self.steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                adjacency[dep].append(step.id)
                in_degree[step.id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.steps):
            raise ValueError("Plan contains a cycle")
        return order

    def ready_steps(self) -> list[PlanStep]:
        """就绪判定：自身 PENDING 且所有前置依赖已 DONE，才可被调度执行。"""
        done_ids = {s.id for s in self.steps if s.status == StepStatus.DONE}
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in done_ids for dep in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps
        )

    def has_failed(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def step_by_id(self, step_id: str) -> PlanStep | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def skip_dependents_of(self, step_id: str) -> list[str]:
        """毒化传播：失败步骤的所有下游依赖递归跳过，已完成或运行中的步骤不受影响。"""
        skipped: list[str] = []
        poisoned = {step_id}
        for step in self.steps:
            if step.status in (StepStatus.DONE, StepStatus.RUNNING):
                continue
            if any(dep in poisoned for dep in step.depends_on):
                step.status = StepStatus.SKIPPED
                skipped.append(step.id)
                poisoned.add(step.id)
        return skipped
