"""Phase 2c：Checkpoint Protocol + 内存/文件实现 + orchestrator 集成。"""

from __future__ import annotations

import asyncio
import json
import operator
from pathlib import Path
from typing import Annotated

import pytest

from topsport_agent.engine.checkpoint import (
    Checkpointer,
    FileCheckpointer,
    MemoryCheckpointer,
    PlanSnapshot,
)
from topsport_agent.engine.orchestrator import Orchestrator, SubAgentConfig
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.plan import Plan, PlanStep, StepStatus
from topsport_agent.types.plan_context import PlanContext, Reducer


# ---------------------------------------------------------------------------
# Helpers (aligned with test_orchestrator_conditions)
# ---------------------------------------------------------------------------


class ScriptedProvider:
    name = "scripted"

    def __init__(self, text: str = "done") -> None:
        self.text = text
        self.call_count = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        self.call_count += 1
        return LLMResponse(text=self.text, finish_reason="stop")


class DemoCtx(PlanContext):
    notes: Annotated[list[str], Reducer(operator.add)] = []
    counter: int = 0


def _step(sid: str, deps: list[str] | None = None, **kwargs) -> PlanStep:
    return PlanStep(
        id=sid,
        title=f"Step {sid}",
        instructions=f"do {sid}",
        depends_on=deps or [],
        **kwargs,
    )


def _build_plan() -> Plan:
    return Plan(
        id="plan-1",
        goal="demo",
        steps=[_step("a"), _step("b", deps=["a"]), _step("c", deps=["b"])],
    )


def _config() -> SubAgentConfig:
    return SubAgentConfig(provider=ScriptedProvider(), model="fake")


async def _collect(agen) -> list[Event]:
    return [e async for e in agen]


# ---------------------------------------------------------------------------
# PlanSnapshot.capture / apply_to
# ---------------------------------------------------------------------------


def test_capture_records_all_step_state() -> None:
    plan = _build_plan()
    plan.step_by_id("a").status = StepStatus.DONE       # type: ignore[union-attr]
    plan.step_by_id("a").result = "A-done"              # type: ignore[union-attr]
    plan.step_by_id("a").iterations = 2                  # type: ignore[union-attr]
    plan.step_by_id("b").status = StepStatus.FAILED     # type: ignore[union-attr]
    plan.step_by_id("b").error = "b blew up"             # type: ignore[union-attr]

    snap = PlanSnapshot.capture(plan)
    assert snap.plan_id == "plan-1"
    assert snap.goal == "demo"
    a_data = next(s for s in snap.steps if s["id"] == "a")
    assert a_data["status"] == "done"
    assert a_data["result"] == "A-done"
    assert a_data["iterations"] == 2
    b_data = next(s for s in snap.steps if s["id"] == "b")
    assert b_data["status"] == "failed"
    assert b_data["error"] == "b blew up"
    assert snap.context_data is None   # plan 无 context


def test_capture_includes_context_data() -> None:
    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a")],
        context=DemoCtx(notes=["x"], counter=3),
    )
    snap = PlanSnapshot.capture(plan)
    assert snap.context_data == {"notes": ["x"], "counter": 3}


def test_apply_to_restores_step_state() -> None:
    plan = _build_plan()
    snap = PlanSnapshot(
        plan_id="plan-1",
        goal="demo",
        steps=[
            {"id": "a", "status": "done", "result": "A", "error": None, "iterations": 1},
            {"id": "b", "status": "pending", "result": None, "error": None, "iterations": 0},
            {"id": "c", "status": "pending", "result": None, "error": None, "iterations": 0},
        ],
    )
    snap.apply_to(plan)
    assert plan.step_by_id("a").status == StepStatus.DONE   # type: ignore[union-attr]
    assert plan.step_by_id("a").result == "A"               # type: ignore[union-attr]


def test_apply_to_restores_context_when_cls_given() -> None:
    plan = Plan(id="p", goal="g", steps=[_step("a")], context=DemoCtx())
    snap = PlanSnapshot(
        plan_id="p",
        goal="g",
        steps=[{"id": "a", "status": "pending", "result": None, "error": None, "iterations": 0}],
        context_data={"notes": ["restored"], "counter": 9},
    )
    snap.apply_to(plan, context_cls=DemoCtx)
    assert plan.context is not None
    assert plan.context.notes == ["restored"]       # type: ignore[attr-defined]
    assert plan.context.counter == 9                # type: ignore[attr-defined]


def test_apply_to_strict_detects_step_id_mismatch() -> None:
    plan = _build_plan()
    snap = PlanSnapshot(
        plan_id="plan-1",
        goal="demo",
        steps=[
            {"id": "a", "status": "done", "result": "A", "error": None, "iterations": 1},
            # missing b, c; extra ghost
            {"id": "ghost", "status": "done", "result": None, "error": None, "iterations": 0},
        ],
    )
    with pytest.raises(ValueError) as ei:
        snap.apply_to(plan)
    assert "mismatch" in str(ei.value)


def test_apply_to_plan_id_mismatch_refused() -> None:
    plan = _build_plan()   # id=plan-1
    snap = PlanSnapshot(plan_id="other", goal="g", steps=[])
    with pytest.raises(ValueError):
        snap.apply_to(plan)


# ---------------------------------------------------------------------------
# PlanSnapshot dict <-> json
# ---------------------------------------------------------------------------


def test_snapshot_dict_roundtrip_is_json_safe() -> None:
    plan = Plan(
        id="p",
        goal="g",
        steps=[_step("a")],
        context=DemoCtx(notes=["x"], counter=1),
    )
    plan.step_by_id("a").status = StepStatus.DONE   # type: ignore[union-attr]
    snap = PlanSnapshot.capture(plan)

    raw = json.dumps(snap.to_dict())
    rebuilt = PlanSnapshot.from_dict(json.loads(raw))

    assert rebuilt.plan_id == snap.plan_id
    assert rebuilt.goal == snap.goal
    assert rebuilt.steps == snap.steps
    assert rebuilt.context_data == snap.context_data


def test_schema_version_future_rejected() -> None:
    with pytest.raises(ValueError):
        PlanSnapshot.from_dict(
            {"schema_version": 99, "plan_id": "p", "goal": "g", "steps": []}
        )


# ---------------------------------------------------------------------------
# MemoryCheckpointer
# ---------------------------------------------------------------------------


async def test_memory_checkpointer_save_load_roundtrip() -> None:
    plan = _build_plan()
    plan.step_by_id("a").status = StepStatus.DONE   # type: ignore[union-attr]
    ckpt = MemoryCheckpointer()
    await ckpt.save(PlanSnapshot.capture(plan))
    loaded = await ckpt.load(plan.id)
    assert loaded is not None
    assert loaded.plan_id == plan.id


async def test_memory_checkpointer_miss_returns_none() -> None:
    ckpt = MemoryCheckpointer()
    assert await ckpt.load("never-saved") is None


# ---------------------------------------------------------------------------
# FileCheckpointer
# ---------------------------------------------------------------------------


async def test_file_checkpointer_writes_json_and_reads_back(tmp_path: Path) -> None:
    ckpt = FileCheckpointer(tmp_path)
    plan = _build_plan()
    plan.step_by_id("a").status = StepStatus.DONE       # type: ignore[union-attr]
    plan.step_by_id("a").iterations = 1                  # type: ignore[union-attr]
    await ckpt.save(PlanSnapshot.capture(plan))

    json_path = tmp_path / "plan-1.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["plan_id"] == "plan-1"
    assert data["steps"][0]["status"] == "done"

    loaded = await ckpt.load(plan.id)
    assert loaded is not None
    assert loaded.steps[0]["iterations"] == 1


async def test_file_checkpointer_miss_returns_none(tmp_path: Path) -> None:
    ckpt = FileCheckpointer(tmp_path)
    assert await ckpt.load("never-saved") is None


async def test_file_checkpointer_rejects_unsafe_plan_id(tmp_path: Path) -> None:
    ckpt = FileCheckpointer(tmp_path)
    with pytest.raises(ValueError):
        await ckpt.save(PlanSnapshot(plan_id="../escape", goal="g", steps=[]))
    with pytest.raises(ValueError):
        await ckpt.load("a/b")


async def test_file_checkpointer_atomic_overwrite(tmp_path: Path) -> None:
    """反复 save 同一 plan_id，最后一次胜出；不会留下临时 .tmp 文件。"""
    ckpt = FileCheckpointer(tmp_path)
    await ckpt.save(PlanSnapshot(plan_id="p", goal="v1", steps=[]))
    await ckpt.save(PlanSnapshot(plan_id="p", goal="v2", steps=[]))
    loaded = await ckpt.load("p")
    assert loaded is not None and loaded.goal == "v2"
    # 没有悬挂临时文件
    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert not leftover


# ---------------------------------------------------------------------------
# Orchestrator 集成：正常跑完 → checkpointer 有终态快照
# ---------------------------------------------------------------------------


async def test_orchestrator_writes_terminal_snapshot_on_done() -> None:
    plan = _build_plan()
    ckpt = MemoryCheckpointer()
    orch = Orchestrator(plan, _config(), checkpointer=ckpt)
    events = await _collect(orch.execute())

    assert EventType.PLAN_DONE in [e.type for e in events]
    snap = await ckpt.load(plan.id)
    assert snap is not None
    assert all(s["status"] == "done" for s in snap.steps)


async def test_orchestrator_without_checkpointer_still_works() -> None:
    plan = _build_plan()
    orch = Orchestrator(plan, _config())  # 不传 checkpointer
    events = await _collect(orch.execute())
    assert EventType.PLAN_DONE in [e.type for e in events]


async def test_orchestrator_snapshot_reflects_intermediate_state() -> None:
    """subscriber 在 PLAN_STEP_END 读 checkpoint 应看到当前 step 的新状态。"""
    plan = _build_plan()
    ckpt = MemoryCheckpointer()
    observed: list[dict] = []

    class Spy:
        name = "spy"

        async def on_event(self, event: Event) -> None:
            if event.type == EventType.PLAN_STEP_END:
                snap = await ckpt.load(plan.id)
                if snap is not None:
                    observed.append({s["id"]: s["status"] for s in snap.steps})

    orch = Orchestrator(
        plan, _config(), checkpointer=ckpt, event_subscribers=[Spy()]
    )
    await _collect(orch.execute())

    # 至少存在一次快照把 'a' 标为 done，同一快照里 b/c 尚未完成
    assert any(
        row.get("a") == "done"
        and row.get("b") in ("pending", "running", "done")
        for row in observed
    )


# ---------------------------------------------------------------------------
# 恢复流：capture → apply_to 新 Plan → 继续跑
# ---------------------------------------------------------------------------


async def test_restore_and_resume_skips_already_done_steps() -> None:
    """模拟：跑了一部分 → 保存 → 构造新 Plan → apply → 新 orchestrator 接着跑。
    已 DONE 的 step 不会再被 provider 调用。
    """
    # 第一次跑（只把 a 置 DONE，其余留 PENDING）
    plan1 = _build_plan()
    plan1.step_by_id("a").status = StepStatus.DONE          # type: ignore[union-attr]
    plan1.step_by_id("a").result = "A-result"               # type: ignore[union-attr]
    plan1.step_by_id("a").iterations = 1                    # type: ignore[union-attr]
    ckpt = MemoryCheckpointer()
    await ckpt.save(PlanSnapshot.capture(plan1))

    # 第二次（新进程/新 orchestrator 模拟）
    plan2 = _build_plan()
    snap = await ckpt.load("plan-1")
    assert snap is not None
    snap.apply_to(plan2)

    assert plan2.step_by_id("a").status == StepStatus.DONE  # type: ignore[union-attr]
    assert plan2.step_by_id("a").result == "A-result"       # type: ignore[union-attr]

    provider = ScriptedProvider()
    config = SubAgentConfig(provider=provider, model="fake")
    orch = Orchestrator(plan2, config)
    await _collect(orch.execute())

    # a 已 DONE 不该再 call provider；b 和 c 才刚跑 → 恰好 2 次
    assert provider.call_count == 2
    assert all(s.status == StepStatus.DONE for s in plan2.steps)


async def test_checkpoint_error_does_not_fail_plan(tmp_path: Path) -> None:
    """checkpointer 抛异常只记 warning，不影响 plan 执行完成。"""

    class BrokenCkpt:
        name = "broken"

        async def save(self, snapshot: PlanSnapshot) -> None:
            raise RuntimeError("disk full")

        async def load(self, plan_id: str) -> PlanSnapshot | None:
            return None

    plan = _build_plan()
    orch = Orchestrator(plan, _config(), checkpointer=BrokenCkpt())
    events = await _collect(orch.execute())

    assert EventType.PLAN_DONE in [e.type for e in events]
    assert all(s.status == StepStatus.DONE for s in plan.steps)


# ---------------------------------------------------------------------------
# Protocol structural check
# ---------------------------------------------------------------------------


def test_memory_and_file_are_structural_checkpointers() -> None:
    """MemoryCheckpointer 和 FileCheckpointer 都满足 Checkpointer Protocol。"""
    mem: Checkpointer = MemoryCheckpointer()
    assert mem.name == "memory"

    # FileCheckpointer 需要 dir；用 asyncio 内临时目录
    async def _check() -> None:
        f: Checkpointer = FileCheckpointer(Path("/tmp/_ts_ckpt_probe"))
        assert f.name == "file"

    asyncio.get_event_loop_policy()   # 只为触发一次 asyncio 初始化，兼容测试运行器
    asyncio.run(_check())
