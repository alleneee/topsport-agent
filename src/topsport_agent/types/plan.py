from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plan_context import PlanContext


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
    # ------------------------------------------------------------------
    # Phase 2b — 条件边 + 受控循环
    # ------------------------------------------------------------------
    # 前置条件：波次开始时 orchestrator 对所有 ready step 统一求值；返回 False → 本轮标 SKIPPED。
    # 函数必须是纯函数（无副作用），因为回跳/重跑场景下同一 context 可能被多次求值。
    condition: Callable[[PlanContext], bool] | None = None
    # 后置条件：step 执行完毕且 context 合并后求值；返回 False → 回 PENDING 且下游重置，
    # 实现 reflect-revise 循环。None 表示"跑一次即完"。
    post_condition: Callable[[PlanContext], bool] | None = None
    # 循环兜底：同一 step 允许被执行的总次数上限（含首次）。默认 1 = 不循环。
    max_iterations: int = 1
    # 运行时计数器：orchestrator 每次启动本 step 时 +=1，达上限仍 post_condition 失败则标 FAILED。
    iterations: int = 0


@dataclass(slots=True)
class Plan:
    id: str
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    # Phase 2b — 共享 context 是可选的；未设置时等同于旧行为（纯 DAG、无 condition 求值、无循环）。
    # 类型运行时是 Any，静态上期望用户传 PlanContext 子类实例；orchestrator 会做 None 检查。
    context: "PlanContext | None" = None

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

    def reset_dependents_of(self, step_id: str) -> list[str]:
        """回跳语义的下游重置：沿 depends_on 反向 BFS 重置所有传递下游。

        语义：
        - DONE / FAILED / PENDING → PENDING，清空 result/error
        - RUNNING / SKIPPED → 保持不变（不中断在跑的，不复活已判决的）
        - iterations 保留（max_iterations 是硬上限，防止循环失控）
        - 未知 step_id → KeyError
        """
        if not any(s.id == step_id for s in self.steps):
            raise KeyError(f"unknown step_id: {step_id}")
        # 反向邻接：dep -> 依赖它的 step 列表
        downstream: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                downstream[dep].append(step.id)
        reset: list[str] = []
        seen: set[str] = set()
        queue = list(downstream[step_id])
        while queue:
            sid = queue.pop(0)
            if sid in seen:
                continue
            seen.add(sid)
            step = self.step_by_id(sid)
            if step is None or step.status in (StepStatus.RUNNING, StepStatus.SKIPPED):
                continue
            step.status = StepStatus.PENDING
            step.result = None
            step.error = None
            reset.append(sid)
            queue.extend(downstream[sid])
        return reset
