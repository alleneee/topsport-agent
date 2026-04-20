from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..llm.provider import LLMProvider
from ..types.events import Event, EventType
from ..types.message import Message, Role
from ..types.plan import Plan, PlanStep, StepDecision, StepStatus
from ..types.session import RunState, Session
from ..types.tool import ToolSpec
from .hooks import ContextProvider, EventSubscriber, FailureHandler, StepConfigurator, ToolSource
from .loop import Engine, EngineConfig

if TYPE_CHECKING:
    # 仅类型标注用；Agent 位于上层包 agent/，运行时循环依赖通过延迟引用避开。
    from ..agent.base import Agent

_logger = logging.getLogger(__name__)


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
    ) -> None:
        """H-A2（Orchestrator 侧）：若给 parent_agent，每个 step 通过
        parent_agent.spawn_child 继承父 Agent 的 context_providers / tool_sources /
        post_step_hooks / event_subscribers —— plan step 自动获得 skills / memory /
        compaction / tracing / metrics 等能力，与 /v1/chat/completions 行为对齐。
        未传 parent_agent 时回退到 SubAgentConfig 的老路径以保持兼容。
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
        engine = Engine(
            provider=config.provider,
            tools=list(config.tools),
            config=EngineConfig(
                model=config.model,
                max_steps=config.max_steps,
                provider_options=config.provider_options,
            ),
            context_providers=list(config.context_providers or []),
            tool_sources=list(config.tool_sources or []),
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

        while not self._plan.is_complete():
            if self._cancel_event.is_set():
                ev = self._event(EventType.CANCELLED, {"plan_id": self._plan.id})
                await self._emit(ev)
                yield ev
                return

            ready = self._plan.ready_steps()
            if not ready:
                break

            step_configs: dict[str, SubAgentConfig] = {}
            for step in ready:
                step_configs[step.id] = await self._configure_step(step)
                step.status = StepStatus.RUNNING
                ev = self._event(
                    EventType.PLAN_STEP_START,
                    {"plan_id": self._plan.id, "step_id": step.id, "title": step.title},
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

            for step in ready:
                ev = self._event(
                    EventType.PLAN_STEP_END,
                    {
                        "plan_id": self._plan.id,
                        "step_id": step.id,
                        "status": step.status.value,
                        "result": step.result,
                        "error": step.error,
                    },
                )
                await self._emit(ev)
                yield ev

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
