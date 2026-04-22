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

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..engine.orchestrator import Orchestrator, SubAgentConfig
from ..tools import file_tools
from ..types.events import EventType
from ..types.plan import Plan, PlanStep, StepDecision
from .auth import require_principal
from .config import ServerConfig
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
async def plan_execute(
    body: PlanExecuteRequest,
    request: Request,
    principal: str = Depends(require_principal),
):
    provider = request.app.state.provider
    server_provider_name: str = request.app.state.provider_name
    cfg: ServerConfig = request.app.state.config
    model = _parse_model(body.model, server_provider_name)

    plan = _build_plan(body)

    # 服务端 clamp：客户端的 max_steps 不得越过运营预算的硬上限
    effective_max_steps = min(body.max_steps, cfg.max_plan_steps)

    # H-A2（Orchestrator 侧）：为这次 plan 执行专门构造一个父 Agent；每个 step 通过
    # parent.spawn_child 继承 Agent 的全部非 tool 能力（skills / memory / compaction
    # / tracing / metrics / plugin hooks），与 /v1/chat/completions 行为对齐。
    agent_factory = getattr(request.app.state, "agent_factory", None)
    parent_agent = agent_factory(provider, model) if agent_factory is not None else None

    # tool 层面：enable_file_tools=False 时子代理不可见 file_tools；即便 parent_agent
    # 的 file_ops 被某些配置打开，这里用空 tools 列表会通过 allowed_tool_names=[] 收窄。
    tools = file_tools() if cfg.enable_file_tools else []

    sub_config = SubAgentConfig(
        provider=provider,
        model=model,
        tools=tools,
        max_steps=effective_max_steps,
    )
    orchestrator = Orchestrator(plan, sub_config, parent_agent=parent_agent)
    _logger.info(
        "plan.execute principal=%s plan_id=%s steps=%d max_steps=%d file_tools=%s parent_agent=%s",
        principal,
        plan.id,
        len(plan.steps),
        effective_max_steps,
        cfg.enable_file_tools,
        parent_agent is not None,
        extra={
            "event": "plan_execute",
            "principal": principal,
            "plan_id": plan.id,
            "step_count": len(plan.steps),
            "max_steps": effective_max_steps,
            "file_tools_enabled": cfg.enable_file_tools,
            "parent_agent": parent_agent is not None,
        },
    )

    # 如果启用 sandbox，plan 的所有子 step session id 都以 f"{plan_id}:" 开头
    # （见 orchestrator._create_session / _spawn_via_parent 的 session_id_prefix），
    # 传给 _stream_plan 以便 finally 时统一 release。
    sandbox_pool = getattr(request.app.state, "sandbox_pool", None)
    sandbox_prefix = f"{plan.id}:" if sandbox_pool is not None else None
    # tenant = principal（与 chat 路径对齐）：预绑定 prefix → tenant，让 plan 子 step
    # 的首次 acquire 走 per_tenant_max_sandboxes 配额，而不是绕过去。
    if sandbox_pool is not None and sandbox_prefix is not None:
        try:
            sandbox_pool.bind_tenant_prefix(sandbox_prefix, principal)
        except Exception:
            _logger.warning(
                "plan.sandbox bind_tenant_prefix failed prefix=%s",
                sandbox_prefix, exc_info=True,
            )

    return StreamingResponse(
        _stream_plan(
            orchestrator, request, parent_agent,
            sandbox_pool=sandbox_pool,
            sandbox_prefix=sandbox_prefix,
        ),
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
    orchestrator: Orchestrator,
    request: Request,
    parent_agent: object | None = None,
    *,
    sandbox_pool: object | None = None,
    sandbox_prefix: str | None = None,
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
        # SEC-005 防御：对外只返回异常类型 + 通用消息，完整 exc 只写服务端日志。
        # 避免 provider 异常 / 栈信息中的 URL、headers、API key 片段外泄给客户端。
        _logger.exception(
            "plan stream failed",
            extra={
                "event": "plan_stream_failed",
                "plan_id": getattr(getattr(orchestrator, "plan", None), "id", None),
            },
        )
        yield sse_event("error", {
            "type": type(exc).__name__,
            "message": "plan execution failed",
        })
    finally:
        if disconnected:
            yield sse_event("cancelled", {"reason": "client_disconnected"})
        # Plan 子 step 创建的沙箱要显式回收，否则要等 sandbox_timeout（30min）
        # 才被 OpenSandbox server 自然清理；多租户下这会长期占用配额。
        if sandbox_pool is not None and sandbox_prefix:
            try:
                released = await sandbox_pool.release_by_prefix(sandbox_prefix)  # type: ignore[attr-defined]
                if released:
                    _logger.info(
                        "plan.sandbox cleanup prefix=%s released=%d",
                        sandbox_prefix, released,
                    )
            except Exception:
                _logger.warning(
                    "plan sandbox cleanup failed prefix=%s", sandbox_prefix,
                    exc_info=True,
                )
        # plan 专用的 parent Agent 本次请求用完必须关闭，释放 browser / plugins 等资源
        if parent_agent is not None:
            try:
                close = getattr(parent_agent, "close", None)
                if close is not None:
                    await close()
            except Exception:
                _logger.warning("failed to close plan parent agent", exc_info=True)
