"""/v1/plan/execute：按 DAG 执行一次 Plan，SSE 推送所有 plan 事件。

事件集（event 名即 EventType 短名）：
    plan_approved / plan_step_start / plan_step_end / plan_step_failed /
    plan_waiting / plan_done / plan_failed / cancelled / error

失败步骤默认走 abort 策略（无人工决策时自动终止）；
客户端断开时触发 orchestrator.cancel()，事件流发送 cancelled 后结束。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..engine.orchestrator import Orchestrator, SubAgentConfig
from ..tools import file_tools
from ..types.events import EventType
from ..types.plan import Plan, PlanStep, StepDecision
from .schemas import PlanExecuteRequest
from .sse import sse_event

_logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_model(model_str: str, server_provider: str) -> str:
    if "/" not in model_str:
        raise HTTPException(400, detail=f"model must be 'provider/name', got {model_str!r}")
    provider, _, name = model_str.partition("/")
    if provider != server_provider:
        raise HTTPException(
            400,
            detail=f"server configured for provider {server_provider!r}, "
            f"got {provider!r}",
        )
    return name


def _build_plan(req: PlanExecuteRequest) -> Plan:
    try:
        return Plan(
            id=req.plan.id,
            goal=req.plan.goal,
            steps=[
                PlanStep(
                    id=s.id,
                    title=s.title,
                    instructions=s.instructions,
                    depends_on=list(s.depends_on),
                )
                for s in req.plan.steps
            ],
        )
    except ValueError as exc:
        raise HTTPException(400, detail=f"invalid plan: {exc}") from exc


@router.post("/v1/plan/execute")
async def plan_execute(body: PlanExecuteRequest, request: Request):
    provider = request.app.state.provider
    server_provider_name: str = request.app.state.provider_name
    model = _parse_model(body.model, server_provider_name)

    plan = _build_plan(body)
    sub_config = SubAgentConfig(
        provider=provider,
        model=model,
        tools=file_tools(),
        max_steps=body.max_steps,
    )
    orchestrator = Orchestrator(plan, sub_config)

    return StreamingResponse(
        _stream_plan(orchestrator, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_EVENT_NAME_MAP = {
    EventType.PLAN_APPROVED: "plan_approved",
    EventType.PLAN_STEP_START: "plan_step_start",
    EventType.PLAN_STEP_END: "plan_step_end",
    EventType.PLAN_STEP_FAILED: "plan_step_failed",
    EventType.PLAN_WAITING: "plan_waiting",
    EventType.PLAN_DONE: "plan_done",
    EventType.PLAN_FAILED: "plan_failed",
    EventType.CANCELLED: "cancelled",
    EventType.ERROR: "error",
}


async def _stream_plan(
    orchestrator: Orchestrator, request: Request
) -> AsyncIterator[str]:
    disconnected = False
    try:
        async for event in orchestrator.execute():
            name = _EVENT_NAME_MAP.get(event.type)
            if name is None:
                continue
            yield sse_event(name, event.payload)

            if event.type == EventType.PLAN_WAITING:
                # 无 human-in-the-loop 接入点时直接 abort，避免挂起。
                orchestrator.provide_decision(StepDecision.ABORT)

            if await request.is_disconnected():
                disconnected = True
                orchestrator.cancel()
                break
    except asyncio.CancelledError:
        orchestrator.cancel()
        raise
    except Exception as exc:
        _logger.exception("plan stream failed")
        yield sse_event("error", {"message": str(exc), "type": type(exc).__name__})
    finally:
        if disconnected:
            yield sse_event("cancelled", {"reason": "client_disconnected"})
