"""Plan 级 checkpoint：每个 step 边界后整体快照 + 按 plan_id 索引持久化。

设计：
- `PlanSnapshot` 只序列化可变状态（status/result/error/iterations/context），
  不序列化 callable（condition/post_condition）、也不序列化 depends_on（结构不变）。
- 恢复时调用方负责构造同样的 Plan 骨架（depends_on/condition 代码里定义），
  再用 `PlanSnapshot.apply_to(plan)` 回填状态。checkpointer 层纯数据，不碰代码。
- `Checkpointer` Protocol：save/load 两个方法，让 Memory/File/Redis 实现同构替换。
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..types.plan import Plan, StepStatus
from ..types.plan_context import PlanContext

__all__ = [
    "Checkpointer",
    "FileCheckpointer",
    "MemoryCheckpointer",
    "PlanSnapshot",
]

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PlanSnapshot:
    """Plan 的最小可序列化状态快照。

    JSON 友好：所有字段都是 primitive/dict，不含 callable。
    `context_data` 是 `PlanContext.snapshot()` 的输出；恢复时由调用方传入 PlanContext 类做 restore。
    """

    plan_id: str
    goal: str
    # 每个 step 只存运行时状态，不存 condition/instructions（代码层面定义）
    steps: list[dict[str, Any]] = field(default_factory=list)
    # None 表示 plan 未配置 PlanContext
    context_data: dict[str, Any] | None = None
    # 版本号，未来兼容/迁移用
    schema_version: int = 1

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @classmethod
    def capture(cls, plan: Plan) -> "PlanSnapshot":
        """从 Plan 对象抽取当前状态做快照。"""
        steps = [
            {
                "id": s.id,
                "status": s.status.value,
                "result": s.result,
                "error": s.error,
                "iterations": s.iterations,
            }
            for s in plan.steps
        ]
        ctx_data = plan.context.snapshot() if plan.context is not None else None
        return cls(
            plan_id=plan.id,
            goal=plan.goal,
            steps=steps,
            context_data=ctx_data,
        )

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def apply_to(
        self,
        plan: Plan,
        *,
        context_cls: type[PlanContext] | None = None,
        strict: bool = True,
    ) -> None:
        """把快照状态回填到已构造好的 Plan 实例（depends_on/condition 由代码重建）。

        - `context_cls`：若快照带 context_data，用此类 restore；不传则跳过（调用方负责）。
        - `strict=True`：Plan 的 step id 集必须和 snapshot 一致（防止代码和快照脱钩）；
          False 时只恢复匹配的 step，忽略多余/缺失（演进期友好但可能掩盖 bug）。
        """
        if self.plan_id != plan.id:
            raise ValueError(
                f"snapshot plan_id {self.plan_id!r} != target plan.id {plan.id!r}"
            )

        snap_by_id = {s["id"]: s for s in self.steps}
        plan_ids = {s.id for s in plan.steps}
        if strict:
            missing = plan_ids - snap_by_id.keys()
            extra = snap_by_id.keys() - plan_ids
            if missing or extra:
                raise ValueError(
                    f"plan/snapshot step id mismatch: missing={sorted(missing)} "
                    f"extra={sorted(extra)}"
                )

        for step in plan.steps:
            payload = snap_by_id.get(step.id)
            if payload is None:
                continue
            step.status = StepStatus(payload["status"])
            step.result = payload.get("result")
            step.error = payload.get("error")
            step.iterations = int(payload.get("iterations", 0))

        if self.context_data is not None:
            if context_cls is None:
                _logger.warning(
                    "snapshot has context_data but no context_cls provided; "
                    "plan.context left untouched"
                )
            else:
                plan.context = context_cls.restore(self.context_data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": self.steps,
            "context_data": self.context_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanSnapshot":
        if data.get("schema_version", 1) != 1:
            raise ValueError(
                f"unsupported snapshot schema_version: {data.get('schema_version')}"
            )
        return cls(
            plan_id=data["plan_id"],
            goal=data["goal"],
            steps=list(data.get("steps", [])),
            context_data=data.get("context_data"),
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Checkpointer(Protocol):
    """Plan 级 checkpoint 后端契约。实现必须是并发安全的（同一 plan_id 串行写）。"""

    name: str

    async def save(self, snapshot: PlanSnapshot) -> None: ...

    async def load(self, plan_id: str) -> PlanSnapshot | None: ...


# ---------------------------------------------------------------------------
# Memory impl
# ---------------------------------------------------------------------------


class MemoryCheckpointer:
    """进程内 dict 实现。测试/单机短期任务首选，重启不留痕。"""

    name = "memory"

    def __init__(self) -> None:
        self._store: dict[str, PlanSnapshot] = {}

    async def save(self, snapshot: PlanSnapshot) -> None:
        self._store[snapshot.plan_id] = snapshot

    async def load(self, plan_id: str) -> PlanSnapshot | None:
        return self._store.get(plan_id)

    def clear(self) -> None:
        """测试/手工清理用；不是 Protocol 的一部分。"""
        self._store.clear()


# ---------------------------------------------------------------------------
# File impl
# ---------------------------------------------------------------------------


class FileCheckpointer:
    """JSON 文件实现。每个 plan 一个 `<plan_id>.json`，原子写（tmp + rename）。

    注意：不做 fsync，也不做跨进程锁。若并发 orchestrator 用同 plan_id 写入，
    依赖文件系统的 rename 原子性（POSIX 保证单文件 rename 原子）即可。
    生产高并发要上 Redis/Postgres，自行实现 Checkpointer Protocol。
    """

    name = "file"

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, plan_id: str) -> Path:
        # plan_id 进入文件名前做基础 sanitize，避免 / .. 逃逸
        if "/" in plan_id or ".." in plan_id or "\x00" in plan_id:
            raise ValueError(f"unsafe plan_id for filesystem: {plan_id!r}")
        return self._base / f"{plan_id}.json"

    async def save(self, snapshot: PlanSnapshot) -> None:
        path = self._path(snapshot.plan_id)
        payload = json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)
        # 原子写：同目录临时文件 → rename，避免读到半写入状态
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self._base),
            prefix=f".{snapshot.plan_id}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

    async def load(self, plan_id: str) -> PlanSnapshot | None:
        path = self._path(plan_id)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        return PlanSnapshot.from_dict(json.loads(raw))


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_checkpoint_hook(
    checkpointer: Checkpointer,
    plan: Plan,
) -> Callable[[], Awaitable[None]]:
    """返回一个"捕获 → 保存"的无参协程闭包；orchestrator 在事件边界调用它。

    设计为闭包而不是方法：调用点不关心具体实现，只关心 `await save_now()`。
    失败按警告吞掉——checkpoint 不能阻塞 plan 执行。
    """

    async def save_now() -> None:
        try:
            snap = PlanSnapshot.capture(plan)
            await checkpointer.save(snap)
        except Exception as exc:
            _logger.warning(
                "checkpoint save failed for plan %r via %r: %r",
                plan.id,
                getattr(checkpointer, "name", "?"),
                exc,
            )

    return save_now
