"""Deprecated /v1/plan/execute — thin forwarder to /v1/chat/completions mode="plan".

Kept for backward compatibility only. The canonical entry point is:

    POST /v1/chat/completions
    {"model": ..., "mode": "plan", "plan": {...}}

Plan execution is an internal Agent strategy (selected via `mode="plan"`),
not a separate subsystem. The Orchestrator and related machinery are
implementation details of Agent.run.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from .auth import require_principal
from .chat import _stream_plan, chat_completions  # re-export for legacy callers
from .schemas import ChatCompletionRequest, PlanExecuteRequest

__all__ = ["router", "_stream_plan"]

router = APIRouter()


@router.post("/v1/plan/execute")
async def plan_execute(
    body: PlanExecuteRequest,
    request: Request,
    principal: str = Depends(require_principal),
):
    """Forward to /v1/chat/completions with mode='plan' semantics.

    Uses plan.id as the OpenAI `user` field to namespace the session —
    mirrors the old endpoint's behavior of fresh-per-plan isolation.
    """
    unified = ChatCompletionRequest(
        model=body.model,
        messages=[],  # plan mode reads from plan.steps, not messages
        stream=True,
        user=f"plan:{body.plan.id}",
        mode="plan",
        plan=body.plan,
    )
    return await chat_completions(unified, request, principal)
