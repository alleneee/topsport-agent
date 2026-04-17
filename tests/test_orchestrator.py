"""Tests for Orchestrator (DAG execution with sub-agents)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from topsport_agent.engine.orchestrator import Orchestrator, SubAgentConfig
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.plan import Plan, PlanStep, StepDecision, StepStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class ScriptedProvider:
    """Returns a canned response for every complete() call."""

    name = "scripted"
    text: str = "done"
    fail: bool = False
    delay: float = 0.0
    call_count: int = 0

    async def complete(self, _request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("provider failure")
        return LLMResponse(text=self.text, finish_reason="stop")


@dataclass
class EventCollector:
    name = "collector"
    events: list[Event] = field(default_factory=list)

    async def on_event(self, event: Event) -> None:
        self.events.append(event)


def _step(id: str, deps: list[str] | None = None) -> PlanStep:
    return PlanStep(
        id=id,
        title=f"Step {id}",
        instructions=f"Do {id}",
        depends_on=deps or [],
    )


def _plan(*steps: PlanStep) -> Plan:
    return Plan(id="test-plan", goal="test goal", steps=list(steps))


def _config(provider: Any = None, **kwargs: Any) -> SubAgentConfig:
    return SubAgentConfig(
        provider=provider or ScriptedProvider(),
        model="fake",
        **kwargs,
    )


async def _collect(agen) -> list[Event]:
    return [event async for event in agen]


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

class TestBasicExecution:
    async def test_single_step_completes(self):
        plan = _plan(_step("a"))
        orch = Orchestrator(plan, _config())
        events = await _collect(orch.execute())

        types = [e.type for e in events]
        assert EventType.PLAN_APPROVED in types
        assert EventType.PLAN_STEP_START in types
        assert EventType.PLAN_STEP_END in types
        assert EventType.PLAN_DONE in types
        assert plan.steps[0].status == StepStatus.DONE

    async def test_linear_chain_executes_in_order(self):
        plan = _plan(_step("a"), _step("b", ["a"]), _step("c", ["b"]))
        orch = Orchestrator(plan, _config())
        events = await _collect(orch.execute())

        step_starts = [
            e.payload["step_id"]
            for e in events
            if e.type == EventType.PLAN_STEP_START
        ]
        assert step_starts == ["a", "b", "c"]
        assert all(s.status == StepStatus.DONE for s in plan.steps)

    async def test_parallel_steps_all_start(self):
        plan = _plan(_step("a"), _step("b"), _step("c"))
        orch = Orchestrator(plan, _config())
        events = await _collect(orch.execute())

        step_starts = [
            e.payload["step_id"]
            for e in events
            if e.type == EventType.PLAN_STEP_START
        ]
        assert set(step_starts) == {"a", "b", "c"}

    async def test_diamond_dag(self):
        plan = _plan(
            _step("a"),
            _step("b", ["a"]),
            _step("c", ["a"]),
            _step("d", ["b", "c"]),
        )
        orch = Orchestrator(plan, _config())
        events = await _collect(orch.execute())

        starts = [
            e.payload["step_id"]
            for e in events
            if e.type == EventType.PLAN_STEP_START
        ]
        assert starts[0] == "a"
        assert set(starts[1:3]) == {"b", "c"}
        assert starts[3] == "d"
        assert plan.is_complete()

    async def test_empty_plan_completes_immediately(self):
        plan = _plan()
        orch = Orchestrator(plan, _config())
        events = await _collect(orch.execute())

        types = [e.type for e in events]
        assert EventType.PLAN_APPROVED in types
        assert EventType.PLAN_DONE in types


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class TestResults:
    async def test_step_result_is_last_assistant_message(self):
        plan = _plan(_step("a"))
        provider = ScriptedProvider(text="the answer is 42")
        orch = Orchestrator(plan, _config(provider))
        await _collect(orch.execute())

        assert plan.steps[0].result == "the answer is 42"

    async def test_plan_done_payload_contains_results(self):
        plan = _plan(_step("a"), _step("b"))
        provider = ScriptedProvider(text="result")
        orch = Orchestrator(plan, _config(provider))
        events = await _collect(orch.execute())

        done_event = next(e for e in events if e.type == EventType.PLAN_DONE)
        assert "a" in done_event.payload["results"]
        assert "b" in done_event.payload["results"]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:
    async def test_step_failure_emits_waiting_event(self):
        plan = _plan(_step("a"))
        provider = ScriptedProvider(fail=True)
        orch = Orchestrator(plan, _config(provider))

        async def provide_abort():
            await asyncio.sleep(0.05)
            orch.provide_decision(StepDecision.ABORT)

        asyncio.create_task(provide_abort())
        events = await _collect(orch.execute())

        types = [e.type for e in events]
        assert EventType.PLAN_STEP_FAILED in types
        assert EventType.PLAN_WAITING in types
        assert EventType.PLAN_FAILED in types

    async def test_abort_skips_remaining_steps(self):
        plan = _plan(_step("a"), _step("b", ["a"]))
        provider = ScriptedProvider(fail=True)
        orch = Orchestrator(plan, _config(provider))

        async def provide_abort():
            await asyncio.sleep(0.05)
            orch.provide_decision(StepDecision.ABORT)

        asyncio.create_task(provide_abort())
        await _collect(orch.execute())

        assert plan.steps[0].status == StepStatus.FAILED
        assert plan.steps[1].status == StepStatus.SKIPPED

    async def test_skip_marks_dependents_skipped(self):
        plan = _plan(_step("a"), _step("b"), _step("c", ["a"]))
        # Make only step a fail: use separate providers
        class SelectiveProvider:
            name = "selective"

            async def complete(self, request: LLMRequest) -> LLMResponse:
                for msg in request.messages:
                    if msg.content and "Do a" in msg.content:
                        raise RuntimeError("fail a")
                return LLMResponse(text="ok", finish_reason="stop")

        orch = Orchestrator(plan, _config(SelectiveProvider()))

        async def provide_skip():
            await asyncio.sleep(0.1)
            orch.provide_decision(StepDecision.SKIP)

        asyncio.create_task(provide_skip())
        await _collect(orch.execute())

        assert plan.steps[0].status == StepStatus.FAILED   # a
        assert plan.steps[1].status == StepStatus.DONE      # b (independent)
        assert plan.steps[2].status == StepStatus.SKIPPED   # c (depends on a)

    async def test_retry_reruns_failed_step(self):
        attempt = 0

        class RetryProvider:
            name = "retry"

            async def complete(self, _request: LLMRequest) -> LLMResponse:
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise RuntimeError("first attempt fails")
                return LLMResponse(text="success", finish_reason="stop")

        plan = _plan(_step("a"))
        orch = Orchestrator(plan, _config(RetryProvider()))

        async def provide_retry():
            await asyncio.sleep(0.05)
            orch.provide_decision(StepDecision.RETRY)

        asyncio.create_task(provide_retry())
        events = await _collect(orch.execute())

        assert plan.steps[0].status == StepStatus.DONE
        assert plan.steps[0].result == "success"
        types = [e.type for e in events]
        assert EventType.PLAN_DONE in types


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

class TestCancel:
    async def test_cancel_before_execute(self):
        plan = _plan(_step("a"))
        orch = Orchestrator(plan, _config())
        orch.cancel()
        events = await _collect(orch.execute())
        assert events == []

    async def test_cancel_during_execution(self):
        plan = _plan(_step("a"))
        provider = ScriptedProvider(delay=1.0)
        orch = Orchestrator(plan, _config(provider))

        async def cancel_soon():
            await asyncio.sleep(0.05)
            orch.cancel()

        asyncio.create_task(cancel_soon())
        events = await _collect(orch.execute())

        types = [e.type for e in events]
        # Step starts but plan gets cancelled on next loop iteration
        assert EventType.PLAN_STEP_START in types


# ---------------------------------------------------------------------------
# Event subscribers
# ---------------------------------------------------------------------------

class TestEventSubscribers:
    async def test_subscribers_receive_plan_events(self):
        plan = _plan(_step("a"))
        collector = EventCollector()
        orch = Orchestrator(plan, _config(), event_subscribers=[collector])
        await _collect(orch.execute())

        sub_types = [e.type for e in collector.events]
        assert EventType.PLAN_APPROVED in sub_types
        assert EventType.PLAN_STEP_START in sub_types
        assert EventType.PLAN_DONE in sub_types

    async def test_broken_subscriber_does_not_crash_orchestrator(self):
        class BrokenSubscriber:
            name = "broken"

            async def on_event(self, _event: Event) -> None:
                raise RuntimeError("subscriber exploded")

        plan = _plan(_step("a"))
        orch = Orchestrator(
            plan, _config(), event_subscribers=[BrokenSubscriber()]
        )
        events = await _collect(orch.execute())

        assert any(e.type == EventType.PLAN_DONE for e in events)


# ---------------------------------------------------------------------------
# StepConfigurator hook
# ---------------------------------------------------------------------------

class TestStepConfigurator:
    async def test_configurator_modifies_tools_per_step(self):
        """StepConfigurator can inject different tools per step."""
        configured_steps: list[str] = []

        class ToolInjector:
            name = "tool-injector"

            async def configure_step(
                self, step: PlanStep, config: SubAgentConfig
            ) -> SubAgentConfig:
                configured_steps.append(step.id)
                return config

        plan = _plan(_step("a"), _step("b"))
        orch = Orchestrator(
            plan, _config(), step_configurators=[ToolInjector()]
        )
        await _collect(orch.execute())

        assert set(configured_steps) == {"a", "b"}
        assert plan.is_complete()

    async def test_configurator_can_swap_provider_per_step(self):
        """Different steps can use different providers via configurator."""

        class PerStepProvider:
            name = "per-step"

            async def configure_step(
                self, step: PlanStep, config: SubAgentConfig
            ) -> SubAgentConfig:
                return SubAgentConfig(
                    provider=ScriptedProvider(text=f"result-{step.id}"),
                    model=config.model,
                    tools=list(config.tools),
                )

        plan = _plan(_step("a"), _step("b"))
        orch = Orchestrator(
            plan, _config(), step_configurators=[PerStepProvider()]
        )
        await _collect(orch.execute())

        assert plan.steps[0].result == "result-a"
        assert plan.steps[1].result == "result-b"

    async def test_configurator_chain_applies_in_order(self):
        """Multiple configurators are applied sequentially."""
        order: list[str] = []

        class First:
            name = "first"

            async def configure_step(
                self, step: PlanStep, config: SubAgentConfig
            ) -> SubAgentConfig:
                order.append("first")
                return config

        class Second:
            name = "second"

            async def configure_step(
                self, step: PlanStep, config: SubAgentConfig
            ) -> SubAgentConfig:
                order.append("second")
                return config

        plan = _plan(_step("a"))
        orch = Orchestrator(
            plan, _config(), step_configurators=[First(), Second()]
        )
        await _collect(orch.execute())

        assert order == ["first", "second"]

    async def test_broken_configurator_does_not_crash(self):
        """A configurator that raises is skipped; execution continues."""

        class Broken:
            name = "broken"

            async def configure_step(
                self, step: PlanStep, config: SubAgentConfig
            ) -> SubAgentConfig:
                raise RuntimeError("configurator exploded")

        plan = _plan(_step("a"))
        orch = Orchestrator(
            plan, _config(), step_configurators=[Broken()]
        )
        events = await _collect(orch.execute())

        assert any(e.type == EventType.PLAN_DONE for e in events)


# ---------------------------------------------------------------------------
# FailureHandler hook
# ---------------------------------------------------------------------------

class TestFailureHandler:
    async def test_failure_handler_auto_retries(self):
        """FailureHandler can auto-retry without manual provide_decision."""
        attempt = 0

        class RetryOnce:
            name = "retry-once"

            async def handle_failure(
                self, plan: Plan, failed_steps: list[PlanStep]
            ) -> StepDecision:
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    return StepDecision.RETRY
                return StepDecision.ABORT

        call_count = 0

        class FlakeProvider:
            name = "flake"

            async def complete(self, _request: LLMRequest) -> LLMResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("transient")
                return LLMResponse(text="ok", finish_reason="stop")

        plan = _plan(_step("a"))
        orch = Orchestrator(
            plan,
            _config(FlakeProvider()),
            failure_handlers=[RetryOnce()],
        )
        events = await _collect(orch.execute())

        assert plan.steps[0].status == StepStatus.DONE
        assert any(e.type == EventType.PLAN_DONE for e in events)

    async def test_failure_handler_auto_skips(self):
        """FailureHandler can auto-skip, marking dependents as skipped."""

        class AlwaysSkip:
            name = "skip"

            async def handle_failure(
                self, plan: Plan, failed_steps: list[PlanStep]
            ) -> StepDecision:
                return StepDecision.SKIP

        plan = _plan(_step("a"), _step("b"), _step("c", ["a"]))

        class FailStepA:
            name = "fail-a"

            async def complete(self, request: LLMRequest) -> LLMResponse:
                for msg in request.messages:
                    if msg.content and "Do a" in msg.content:
                        raise RuntimeError("fail a")
                return LLMResponse(text="ok", finish_reason="stop")

        orch = Orchestrator(
            plan,
            _config(FailStepA()),
            failure_handlers=[AlwaysSkip()],
        )
        await _collect(orch.execute())

        assert plan.steps[0].status == StepStatus.FAILED
        assert plan.steps[1].status == StepStatus.DONE
        assert plan.steps[2].status == StepStatus.SKIPPED

    async def test_failure_handler_aborts(self):
        """FailureHandler returning ABORT stops the plan."""

        class AlwaysAbort:
            name = "abort"

            async def handle_failure(
                self, plan: Plan, failed_steps: list[PlanStep]
            ) -> StepDecision:
                return StepDecision.ABORT

        plan = _plan(_step("a"), _step("b", ["a"]))
        orch = Orchestrator(
            plan,
            _config(ScriptedProvider(fail=True)),
            failure_handlers=[AlwaysAbort()],
        )
        events = await _collect(orch.execute())

        assert plan.steps[0].status == StepStatus.FAILED
        assert plan.steps[1].status == StepStatus.SKIPPED
        assert any(e.type == EventType.PLAN_FAILED for e in events)

    async def test_broken_handler_falls_through_to_provide_decision(self):
        """If handler raises, falls back to manual provide_decision."""

        class BrokenHandler:
            name = "broken"

            async def handle_failure(
                self, plan: Plan, failed_steps: list[PlanStep]
            ) -> StepDecision:
                raise RuntimeError("handler crashed")

        plan = _plan(_step("a"))
        orch = Orchestrator(
            plan,
            _config(ScriptedProvider(fail=True)),
            failure_handlers=[BrokenHandler()],
        )

        async def provide_abort():
            await asyncio.sleep(0.05)
            orch.provide_decision(StepDecision.ABORT)

        asyncio.create_task(provide_abort())
        events = await _collect(orch.execute())

        assert any(e.type == EventType.PLAN_FAILED for e in events)

    async def test_handler_chain_first_wins(self):
        """First handler to return a decision wins; later handlers skipped."""
        calls: list[str] = []

        class First:
            name = "first"

            async def handle_failure(
                self, plan: Plan, failed_steps: list[PlanStep]
            ) -> StepDecision:
                calls.append("first")
                return StepDecision.ABORT

        class Second:
            name = "second"

            async def handle_failure(
                self, plan: Plan, failed_steps: list[PlanStep]
            ) -> StepDecision:
                calls.append("second")
                return StepDecision.RETRY

        plan = _plan(_step("a"))
        orch = Orchestrator(
            plan,
            _config(ScriptedProvider(fail=True)),
            failure_handlers=[First(), Second()],
        )
        events = await _collect(orch.execute())

        assert calls == ["first"]
        assert any(e.type == EventType.PLAN_FAILED for e in events)
