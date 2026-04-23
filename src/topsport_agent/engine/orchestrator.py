from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..llm.provider import LLMProvider
from ..types.events import Event, EventType
from ..types.message import Message, Role
from ..types.plan import Plan, PlanStep, StepDecision, StepStatus
from ..types.session import RunState, Session
from ..types.tool import ToolSpec
from .checkpoint import Checkpointer, build_checkpoint_hook
from .hooks import ContextProvider, EventSubscriber, FailureHandler, StepConfigurator, ToolSource
from .loop import Engine, EngineConfig
from .plan_context_tools import PlanContextBridge, PlanContextToolSource

if TYPE_CHECKING:
    # 仅类型标注用；Agent 位于上层包 agent/，运行时循环依赖通过延迟引用避开。
    from ..agent.base import Agent

_logger = logging.getLogger(__name__)


async def _noop_async() -> None:
    """无 checkpointer 时的占位 hook。"""
    return None


@dataclass(slots=True)
class SubAgentConfig:
    provider: LLMProvider
    model: str
    tools: list[ToolSpec] = field(default_factory=list)
    context_providers: list[ContextProvider] | None = None
    tool_sources: list[ToolSource] | None = None
    max_steps: int = 20
    provider_options: dict[str, Any] | None = None


class Orchestrator:
    """按 DAG 拓扑序执行 Plan，每个 step 由独立的 Engine+Session 子代理承担。"""

    def __init__(
        self,
        plan: Plan,
        agent_config: SubAgentConfig,
        *,
        step_configurators: list[StepConfigurator] | None = None,
        failure_handlers: list[FailureHandler] | None = None,
        event_subscribers: list[EventSubscriber] | None = None,
        parent_agent: Agent | None = None,
        checkpointer: Checkpointer | None = None,
    ) -> None:
        """H-A2（Orchestrator 侧）：若给 parent_agent，每个 step 通过
        parent_agent.spawn_child 继承父 Agent 的 context_providers / tool_sources /
        post_step_hooks / event_subscribers —— plan step 自动获得 skills / memory /
        compaction / tracing / metrics 等能力，与 /v1/chat/completions 行为对齐。
        未传 parent_agent 时回退到 SubAgentConfig 的老路径以保持兼容。

        Phase 2c: `checkpointer` 给定后，在每个 step 边界事件后整体快照 plan。
        None 时退化为 noop（与现有行为完全一致）。
        """
        self._plan = plan
        self._config = agent_config
        self._parent_agent = parent_agent
        self._step_configurators = list(step_configurators or [])
        self._failure_handlers = list(failure_handlers or [])
        self._event_subscribers = list(event_subscribers or [])
        self._cancel_event = asyncio.Event()
        # 外部调用者通过 provide_decision 写入决策并 set event，阻塞中的 _wait_for_decision 随即解除。
        self._decision_event = asyncio.Event()
        self._pending_decision: StepDecision | None = None
        # 追踪存活的子引擎，cancel 时需要逐个传播。
        self._sub_engines: dict[str, Engine] = {}
        # 无 checkpointer → noop；不在 execute() 里做 if ckpt is None 分支，保持调用点干净。
        self._checkpoint: Callable[[], Awaitable[None]] = (
            build_checkpoint_hook(checkpointer, plan) if checkpointer else _noop_async
        )
        # Plan 层共享 context 存在时，给 sub-agent 挂上 plan_context_read / plan_context_merge 工具
        # 让 LLM 能显式读写共享状态（post_condition 判定就靠这些写入）。未配 context 则不注入，
        # 避免 sub-agent 误用空工具。
        self._context_bridge: PlanContextBridge | None = (
            PlanContextBridge(plan) if plan.context is not None else None
        )

    @property
    def plan(self) -> Plan:
        return self._plan

    def cancel(self) -> None:
        # 取消向下传播：编排器取消 -> 所有存活子引擎取消。
        self._cancel_event.set()
        for engine in self._sub_engines.values():
            engine.cancel()

    def provide_decision(self, decision: StepDecision) -> None:
        self._pending_decision = decision
        self._decision_event.set()

    async def _emit(self, event: Event) -> None:
        for subscriber in self._event_subscribers:
            try:
                await subscriber.on_event(event)
            except Exception as exc:
                _logger.warning(
                    "subscriber %r failed: %r",
                    getattr(subscriber, "name", "?"),
                    exc,
                )

    def _event(self, etype: EventType, payload: dict[str, Any]) -> Event:
        return Event(type=etype, session_id=self._plan.id, payload=payload)

    async def _try_failure_handlers(
        self, failed: list[PlanStep]
    ) -> StepDecision | None:
        # 依次尝试自动决策；某个 handler 成功返回即短路，全部失败则交由外部人工决策。
        for handler in self._failure_handlers:
            try:
                return await handler.handle_failure(self._plan, failed)
            except Exception as exc:
                _logger.warning(
                    "failure handler %r raised: %r",
                    getattr(handler, "name", "?"),
                    exc,
                )
        return None

    async def _wait_for_decision(self) -> StepDecision:
        # 阻塞直到外部调用者通过 provide_decision 注入决策，避免自动循环失败步骤。
        self._decision_event.clear()
        self._pending_decision = None
        await self._decision_event.wait()
        assert self._pending_decision is not None
        return self._pending_decision

    def _create_engine(self, step: PlanStep, config: SubAgentConfig) -> Engine:
        # 每个 step 拿到完全隔离的 Engine+Session，不共享任何可变状态。
        # 事件订阅者直接透传给子引擎，子引擎事件不经过编排器中转。
        sources: list[ToolSource] = list(config.tool_sources or [])
        if self._context_bridge is not None:
            sources.append(PlanContextToolSource(self._context_bridge))
        engine = Engine(
            provider=config.provider,
            tools=list(config.tools),
            config=EngineConfig(
                model=config.model,
                max_steps=config.max_steps,
                provider_options=config.provider_options,
            ),
            context_providers=list(config.context_providers or []),
            tool_sources=sources,
            event_subscribers=list(self._event_subscribers),
        )
        self._sub_engines[step.id] = engine
        return engine

    async def _configure_step(self, step: PlanStep) -> SubAgentConfig:
        # 配置器链式执行，允许按 step 定制模型/工具/上下文。某个配置器异常时跳过，用上一个有效配置兜底。
        config = self._config
        for configurator in self._step_configurators:
            try:
                config = await configurator.configure_step(step, config)
            except Exception as exc:
                _logger.warning(
                    "step configurator %r failed for step %r: %r",
                    getattr(configurator, "name", "?"),
                    step.id,
                    exc,
                )
        return config

    def _create_session(self, step: PlanStep) -> Session:
        session = Session(
            id=f"{self._plan.id}:{step.id}",
            system_prompt=(
                "You are a sub-agent executing one step of a larger plan. "
                "Complete the task described below."
            ),
            goal=step.title,
        )
        session.messages.append(
            Message(
                role=Role.USER,
                content=f"## Task: {step.title}\n\n{step.instructions}",
            )
        )
        return session

    @staticmethod
    def _extract_result(session: Session) -> str | None:
        for msg in reversed(session.messages):
            if msg.role == Role.ASSISTANT and msg.content:
                return msg.content
        return None

    async def _run_step(self, step: PlanStep, config: SubAgentConfig) -> None:
        # 跑完整个子代理循环，最终把引擎结果映射回 step 状态。引擎结束后立即从存活表移除。
        # iterations 在执行前 +=1：哪怕 sub-agent 异常也算一次尝试，避免异常路径绕过 max_iterations。
        step.iterations += 1
        try:
            if self._parent_agent is not None:
                session, engine = await self._spawn_via_parent(step, config)
            else:
                engine = self._create_engine(step, config)
                session = self._create_session(step)
            self._sub_engines[step.id] = engine

            async for _ in engine.run(session):
                pass

            self._sub_engines.pop(step.id, None)

            if session.state == RunState.DONE:
                step.status = StepStatus.DONE
                step.result = self._extract_result(session)
            else:
                step.status = StepStatus.FAILED
                step.error = f"Sub-agent ended in state: {session.state.value}"
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.error = f"{type(exc).__name__}: {exc}"
            self._sub_engines.pop(step.id, None)

    async def _spawn_via_parent(
        self, step: PlanStep, config: SubAgentConfig
    ) -> tuple[Session, Engine]:
        """走 parent_agent.spawn_child 路径：继承所有非 tool 能力。
        config.tools 如果为空，仍然使用父 Agent 的全部工具；非空则按名字过滤（与
        spawn_agent 的 allowed_tools 语义一致）。
        """
        assert self._parent_agent is not None
        allowed = [t.name for t in config.tools] if config.tools else None
        session, engine = await self._parent_agent.spawn_child(
            model=config.model,
            system_prompt=(
                "You are a sub-agent executing one step of a larger plan. "
                "Complete the task described below."
            ),
            task=f"## Task: {step.title}\n\n{step.instructions}",
            allowed_tool_names=allowed,
            session_id_prefix=f"{self._plan.id}:{step.id}",
        )
        session.goal = step.title
        # 把 orchestrator 自己订阅者也挂到子引擎（和旧路径一致）
        for sub in self._event_subscribers:
            engine.add_event_subscriber(sub)
        # plan_context bridge 走 add_tool_source（spawn_child 已构造完 engine，无法走构造参数）
        if self._context_bridge is not None:
            engine.add_tool_source(PlanContextToolSource(self._context_bridge))
        return session, engine

    async def execute(self) -> AsyncIterator[Event]:
        # 主循环：按波次推进 DAG，每波取出所有无前置依赖的 ready step 并行执行。
        if self._cancel_event.is_set():
            return

        ev = self._event(
            EventType.PLAN_APPROVED,
            {
                "plan_id": self._plan.id,
                "goal": self._plan.goal,
                "step_count": len(self._plan.steps),
            },
        )
        await self._emit(ev)
        yield ev
        # Phase 2c: 起点快照，让 load() 能拿到"已批准但未开跑"的初始态
        await self._checkpoint()

        while not self._plan.is_complete():
            if self._cancel_event.is_set():
                ev = self._event(EventType.CANCELLED, {"plan_id": self._plan.id})
                await self._emit(ev)
                yield ev
                return

            ready = self._plan.ready_steps()
            if not ready:
                break

            # Phase 2b: condition 过滤。波次开始时对本波 ready step 统一求值（context 快照），
            # 同波并发 step 互不影响彼此的 condition 输入——可预测、调试友好。
            # condition False → 标 SKIPPED，不进入本波执行。
            filtered_ready: list[PlanStep] = []
            ctx = self._plan.context
            for step in ready:
                if step.condition is not None and ctx is not None:
                    try:
                        passed = bool(step.condition(ctx))
                    except Exception as exc:
                        # condition 抛错按"未满足"处理，标 SKIPPED 并记录 error，避免整个 plan 崩。
                        step.status = StepStatus.SKIPPED
                        step.error = f"condition raised {type(exc).__name__}: {exc}"
                        skip_ev = self._event(
                            EventType.PLAN_STEP_SKIPPED,
                            {
                                "plan_id": self._plan.id,
                                "step_id": step.id,
                                "reason": "condition_error",
                                "error": step.error,
                            },
                        )
                        await self._emit(skip_ev)
                        yield skip_ev
                        await self._checkpoint()
                        continue
                    if not passed:
                        step.status = StepStatus.SKIPPED
                        skip_ev = self._event(
                            EventType.PLAN_STEP_SKIPPED,
                            {
                                "plan_id": self._plan.id,
                                "step_id": step.id,
                                "reason": "condition_false",
                            },
                        )
                        await self._emit(skip_ev)
                        yield skip_ev
                        await self._checkpoint()
                        continue
                filtered_ready.append(step)

            # 整波都被 condition 过滤掉：下一轮 while 会重新检查完成态；若所有剩余 pending
            # 都被 condition 过滤，is_complete() 为 true（SKIPPED 计入终态），正常结束。
            if not filtered_ready:
                continue

            ready = filtered_ready

            step_configs: dict[str, SubAgentConfig] = {}
            for step in ready:
                step_configs[step.id] = await self._configure_step(step)
                step.status = StepStatus.RUNNING
                ev = self._event(
                    EventType.PLAN_STEP_START,
                    {
                        "plan_id": self._plan.id,
                        "step_id": step.id,
                        "title": step.title,
                        "iteration": step.iterations + 1,  # _run_step 里会 +=1
                    },
                )
                await self._emit(ev)
                yield ev

            # 同一波次的 step 通过 gather 并行跑，互不阻塞。
            await asyncio.gather(
                *[self._run_step(step, step_configs[step.id]) for step in ready]
            )

            # gather 返回后先检查取消，再处理失败——防止失败步骤尝试等待人工决策时与取消信号死锁。
            if self._cancel_event.is_set():
                ev = self._event(EventType.CANCELLED, {"plan_id": self._plan.id})
                await self._emit(ev)
                yield ev
                return

            # Phase 2b: post_condition 回跳判定。只对本波 DONE 且带 post_condition 的 step 求值。
            # 返回 False → 本 step 标回 PENDING、重置下游、发 PLAN_STEP_LOOP；
            # 达到 max_iterations 仍 False → 标 FAILED（走现有失败决策流程）。
            if ctx is not None:
                for step in ready:
                    if step.status != StepStatus.DONE or step.post_condition is None:
                        continue
                    try:
                        satisfied = bool(step.post_condition(ctx))
                    except Exception as exc:
                        step.status = StepStatus.FAILED
                        step.error = (
                            f"post_condition raised {type(exc).__name__}: {exc}"
                        )
                        continue
                    if satisfied:
                        continue
                    # 不满足 → 判是否还能再跑
                    if step.iterations >= step.max_iterations:
                        step.status = StepStatus.FAILED
                        step.error = (
                            f"post_condition not satisfied after "
                            f"{step.iterations} iteration(s) (max={step.max_iterations})"
                        )
                        continue
                    # 回跳：本 step 回 PENDING + 重置传递下游，下轮 while 再次 ready
                    reset_ids = self._plan.reset_dependents_of(step.id)
                    step.status = StepStatus.PENDING
                    step.result = None
                    step.error = None
                    loop_ev = self._event(
                        EventType.PLAN_STEP_LOOP,
                        {
                            "plan_id": self._plan.id,
                            "step_id": step.id,
                            "iteration": step.iterations,
                            "max_iterations": step.max_iterations,
                            "reset_dependents": reset_ids,
                        },
                    )
                    await self._emit(loop_ev)
                    yield loop_ev
                    await self._checkpoint()

            for step in ready:
                # 被 LOOP 重置回 PENDING 的 step 不发 PLAN_STEP_END（它还没 "end"），
                # 其它状态（DONE/FAILED/SKIPPED）都要发一次 END 让订阅者记账。
                if step.status == StepStatus.PENDING:
                    continue
                ev = self._event(
                    EventType.PLAN_STEP_END,
                    {
                        "plan_id": self._plan.id,
                        "step_id": step.id,
                        "status": step.status.value,
                        "result": step.result,
                        "error": step.error,
                        "iterations": step.iterations,
                    },
                )
                await self._emit(ev)
                yield ev

            # 每波 step 全部 END 后做一次波次级快照（比 per-step 少写放大，仍足够细粒度）
            await self._checkpoint()

            failed = [s for s in ready if s.status == StepStatus.FAILED]
            if failed:
                ev = self._event(
                    EventType.PLAN_STEP_FAILED,
                    {
                        "plan_id": self._plan.id,
                        "failed_steps": [
                            {"id": s.id, "error": s.error} for s in failed
                        ],
                    },
                )
                await self._emit(ev)
                yield ev

                # 失败路径：先尝试 handler 自动决策，全部兜不住再 yield PLAN_WAITING 让外部人工介入。
                decision = await self._try_failure_handlers(failed)
                if decision is None:
                    wait_ev = self._event(
                        EventType.PLAN_WAITING,
                        {
                            "plan_id": self._plan.id,
                            "options": ["retry", "skip", "abort"],
                        },
                    )
                    await self._emit(wait_ev)
                    yield wait_ev
                    decision = await self._wait_for_decision()

                # abort: 剩余 pending 全部标 skipped，整个 plan 终止。
                # skip: 跳过失败步骤并级联跳过其下游依赖。
                # retry: 把失败步骤重置为 pending，下一轮 while 循环会重新拾取。
                if decision == StepDecision.ABORT:
                    for s in self._plan.steps:
                        if s.status == StepStatus.PENDING:
                            s.status = StepStatus.SKIPPED
                    ev = self._event(
                        EventType.PLAN_FAILED,
                        {"plan_id": self._plan.id, "reason": "aborted"},
                    )
                    await self._emit(ev)
                    yield ev
                    await self._checkpoint()
                    return
                elif decision == StepDecision.SKIP:
                    for s in failed:
                        self._plan.skip_dependents_of(s.id)
                elif decision == StepDecision.RETRY:
                    for s in failed:
                        s.status = StepStatus.PENDING
                        s.error = None

        # 正常结束 vs DAG 卡死：is_complete 为 true 说明所有 step 已终态；否则是无 ready step 的死锁。
        if self._plan.is_complete():
            ev = self._event(
                EventType.PLAN_DONE,
                {
                    "plan_id": self._plan.id,
                    "results": {
                        s.id: s.result
                        for s in self._plan.steps
                        if s.status == StepStatus.DONE
                    },
                },
            )
            await self._emit(ev)
            yield ev
        else:
            ev = self._event(
                EventType.PLAN_FAILED,
                {"plan_id": self._plan.id, "reason": "no_ready_steps"},
            )
            await self._emit(ev)
            yield ev
        # 终态快照：无论 DONE / FAILED / no_ready_steps，存一份给下游 observer 消费。
        await self._checkpoint()
