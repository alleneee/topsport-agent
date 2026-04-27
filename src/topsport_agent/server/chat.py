"""/v1/chat/completions：OpenAI 兼容，JSON 与 SSE 两种响应形态。

语义约定：
- 服务端有状态，`user` 字段当作 session_id，未带则生成新 id
- 客户端发来的 messages 中，只取最后一条 role=user 作为本轮新输入
- 历史消息由服务端 session 维护，不采信客户端 history
- SSE 事件集: 只产出 OpenAI 标准 delta + [DONE]，不透传工具事件
- 客户端断开：generator 取消 -> agent.cancel() 传播到 LLM 调用
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

# Sentinel value returned by `_next_event` when the agent iterator is
# exhausted — lets us detect end-of-stream without StopAsyncIteration
# propagating across the asyncio.wait boundary.
_END = object()

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..types.events import EventType
from ..types.message import Role
from ..types.plan import Plan, PlanStep
from ..types.plan_context_kv import KVPlanContext
from .auth import namespace_session_id, require_principal
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    PlanSchema,
)
from .sessions import SessionEntry, SessionStore
from .sse import SSE_DONE_LINE, make_chat_chunk, sse_data, sse_event

_logger = logging.getLogger(__name__)

router = APIRouter()


def _extract_last_user_message(messages: list) -> str | None:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return None


def _extract_system_prompt(body) -> str | None:
    """Pick the request-level system prompt override.

    Precedence: body.system > first messages[] entry with role=system.
    Returns None if neither is set; caller then keeps the session default.
    """
    if body.system:
        return body.system
    for m in body.messages:
        if m.role == "system" and m.content:
            return m.content
    return None


def _parse_model(model_str: str) -> tuple[str, str]:
    if "/" not in model_str:
        raise HTTPException(400, detail=f"model must be 'provider/name', got {model_str!r}")
    provider, _, name = model_str.partition("/")
    provider = provider.strip().lower()
    name = name.strip()
    if provider not in ("anthropic", "openai"):
        raise HTTPException(400, detail=f"unknown provider {provider!r}")
    if not name:
        raise HTTPException(400, detail="model name is empty")
    return provider, name


def _build_plan(schema: PlanSchema) -> Plan:
    """Construct a Plan with a default shared KV context.

    Every HTTP-initiated plan gets KVPlanContext() so steps can use the
    auto-mounted plan_context_read / plan_context_merge tools without the
    client having to declare a context shape.
    """
    try:
        return Plan(
            id=schema.id,
            goal=schema.goal,
            steps=[
                PlanStep(
                    id=s.id,
                    title=s.title,
                    instructions=s.instructions,
                    depends_on=list(s.depends_on),
                    max_iterations=s.max_iterations,
                )
                for s in schema.steps
            ],
            context=KVPlanContext(),
        )
    except ValueError as exc:
        raise HTTPException(400, detail=f"invalid plan: {exc}") from exc


# Plan event → SSE event name (flat list; any new EventType adds one entry here).
_PLAN_EVENT_NAME = {
    EventType.PLAN_APPROVED: "plan_approved",
    EventType.PLAN_STEP_START: "plan_step_start",
    EventType.PLAN_STEP_END: "plan_step_end",
    EventType.PLAN_STEP_FAILED: "plan_step_failed",
    EventType.PLAN_STEP_SKIPPED: "plan_step_skipped",
    EventType.PLAN_WAITING: "plan_waiting",
    EventType.PLAN_DONE: "plan_done",
    EventType.PLAN_FAILED: "plan_failed",
    EventType.CANCELLED: "cancelled",
    EventType.ERROR: "error",
}


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    principal: str = Depends(require_principal),
):
    store: SessionStore = request.app.state.session_store
    server_provider: str = request.app.state.provider_name

    req_provider, model_name = _parse_model(body.model)
    if req_provider != server_provider:
        raise HTTPException(
            400,
            detail=f"server configured for provider {server_provider!r}, "
            f"got {req_provider!r} in request",
        )

    # principal 前缀隔离：不同 principal 即便传同一个 body.user 也命中不同 session
    session_key = namespace_session_id(principal, body.user)
    # tenant = principal（最简映射）：sandbox / per-tenant quota / 审计都按 principal 维度做。
    _, entry, _ = await store.get_or_create(
        session_key, model_name, tenant_id=principal, principal=principal,
    )

    # 请求级 system prompt 覆盖：持久到 session，后续轮次延用。
    override = _extract_system_prompt(body)
    if override is not None and override != entry.session.system_prompt:
        entry.session.system_prompt = override

    # Mode dispatch — Plan 不是独立端点，是 Agent 的一种执行策略。
    if body.mode == "plan":
        if body.plan is None:
            raise HTTPException(
                400, detail="mode='plan' requires 'plan' field"
            )
        cfg = request.app.state.config
        max_plan_steps = getattr(cfg, "max_plan_steps", 20)
        if len(body.plan.steps) > max_plan_steps:
            raise HTTPException(
                400,
                detail=f"plan has {len(body.plan.steps)} steps, exceeds server limit {max_plan_steps}",
            )
        plan_obj = _build_plan(body.plan)
        return StreamingResponse(
            _stream_plan(entry, plan_obj, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Default: mode == "react"
    user_input = _extract_last_user_message(body.messages)
    if not user_input:
        raise HTTPException(400, detail="messages must contain at least one user message")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.stream:
        return StreamingResponse(
            _stream_chat(entry, user_input, chat_id, body.model, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(await _complete_chat(entry, user_input, chat_id, body.model))


async def _stream_plan(
    entry: SessionEntry,
    plan: Plan,
    request: Request,
) -> AsyncIterator[str]:
    """Stream Plan execution events through the unified agent.run entry point.

    Agent.run(mode="plan", plan=...) dispatches to the internal Orchestrator
    strategy. Failed steps follow the Orchestrator's default abort policy;
    client disconnect cancels the entire plan.

    Sandbox lifecycle: when the app has a sandbox_pool, bind the plan-id
    prefix to this principal so every sub-step's sandbox acquire counts
    against the same tenant quota, and release them together on plan end.
    """
    sandbox_pool = getattr(request.app.state, "sandbox_pool", None)
    sandbox_prefix = f"{plan.id}:" if sandbox_pool is not None else None
    principal = entry.session.principal or entry.session.tenant_id or "anonymous"
    if sandbox_pool is not None and sandbox_prefix is not None:
        try:
            sandbox_pool.bind_tenant_prefix(sandbox_prefix, principal)
        except Exception:
            _logger.warning(
                "plan.sandbox bind_tenant_prefix failed prefix=%s",
                sandbox_prefix, exc_info=True,
            )

    async with entry.lock:
        try:
            async for event in entry.agent.run(
                session=entry.session, mode="plan", plan=plan,
            ):
                if await request.is_disconnected():
                    entry.agent.cancel()
                    break
                name = _PLAN_EVENT_NAME.get(event.type)
                if name is not None:
                    yield sse_event(name, event.payload)
        except asyncio.CancelledError:
            entry.agent.cancel()
            raise
        except Exception as exc:
            # SEC-005: 不能把 str(exc) 回给客户端（可能含 API key / 内网 URL 等）。
            # 只暴露异常类型；详细信息只进服务端日志。
            _logger.exception(
                "plan stream failed",
                extra={
                    "event": "plan_stream_failed",
                    "session_id": entry.session.id,
                    "plan_id": plan.id,
                },
            )
            yield sse_event(
                "error",
                {"message": "plan execution failed", "type": type(exc).__name__},
            )
        finally:
            if sandbox_pool is not None and sandbox_prefix is not None:
                try:
                    released = await sandbox_pool.release_by_prefix(sandbox_prefix)
                    if released:
                        _logger.info(
                            "plan.sandbox cleanup prefix=%s released=%d",
                            sandbox_prefix, released,
                        )
                except Exception:
                    _logger.warning(
                        "plan sandbox cleanup failed prefix=%s",
                        sandbox_prefix, exc_info=True,
                    )


async def _complete_chat(
    entry: SessionEntry, user_input: str, chat_id: str, model: str
) -> dict:
    async with entry.lock:
        errors: list[str] = []
        async for event in entry.agent.run(user_input, entry.session):
            if event.type == EventType.ERROR:
                errors.append(
                    f"{event.payload.get('kind')}: {event.payload.get('message')}"
                )

        final_text = ""
        for msg in reversed(entry.session.messages):
            if msg.role == Role.ASSISTANT and msg.content:
                final_text = msg.content
                break

    if errors and not final_text:
        raise HTTPException(500, detail={"errors": errors})

    resp = ChatCompletionResponse(
        id=chat_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=final_text),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(),
    )
    return resp.model_dump()


async def _stream_chat(
    entry: SessionEntry,
    user_input: str,
    chat_id: str,
    model: str,
    request: Request,
) -> AsyncIterator[str]:
    async with entry.lock:
        # 首帧：带 role 的空 delta，OpenAI SSE 协议要求
        yield sse_data(make_chat_chunk(chat_id, model, delta={"role": "assistant"}))

        saw_delta = False
        baseline_msg_count = len(entry.session.messages)

        # Server-initiated elicitation: drain pending requests for this
        # session ASAP after they arrive. Two wake-up sources merged via
        # asyncio.wait(FIRST_COMPLETED):
        #   1. agent.run iterator's next event (chat content / tool / error)
        #   2. broker's per-session signal (set whenever new elicitation
        #      arrives — wakes us up even during long LLM thinking
        #      where no agent events would surface).
        # Without (2), elicitation frames would queue up and time out
        # while the LLM is mid-call.
        broker = getattr(request.app.state, "elicitation_broker", None)
        sid = entry.session.id
        elicit_signal = broker.signal_for(sid) if broker is not None else None

        async def _drain_elicitations() -> AsyncIterator[str]:
            if broker is None:
                return
            try:
                pending = await broker.pending_for_session(sid)
            except Exception as exc:
                _logger.warning(
                    "elicitation drain failed for session=%s: %r",
                    sid, exc, exc_info=True,
                )
                return
            for eid, req in pending:
                yield sse_event("elicitation", {
                    "id": eid,
                    "message": req.message,
                    "mode": req.mode,
                    "requested_schema": req.requested_schema,
                    "url": req.url,
                })

        try:
            agent_iter = entry.agent.run(user_input, entry.session).__aiter__()

            async def _next_event() -> Any:
                """Wrapper makes the next-event read a real coroutine
                (Pyright requires create_task input to be Coroutine, not
                bare Awaitable). Returns the sentinel `_END` when iter
                is exhausted so we can detect that without exception
                propagation across the asyncio.wait boundary."""
                try:
                    return await agent_iter.__anext__()
                except StopAsyncIteration:
                    return _END

            agent_task: asyncio.Task | None = asyncio.create_task(_next_event())
            signal_task: asyncio.Task | None = (
                asyncio.create_task(elicit_signal.wait())
                if elicit_signal is not None else None
            )

            try:
                while agent_task is not None:
                    if await request.is_disconnected():
                        entry.agent.cancel()
                        break

                    waiting: list[asyncio.Task] = [agent_task]
                    if signal_task is not None:
                        waiting.append(signal_task)

                    done, _pending = await asyncio.wait(
                        waiting, return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Process signal first so any new elicitation frame
                    # is interleaved before the next agent event.
                    if signal_task is not None and signal_task in done:
                        # Clear before draining so a concurrent push
                        # re-sets and triggers another wake-up.
                        if elicit_signal is not None:
                            elicit_signal.clear()
                        async for frame in _drain_elicitations():
                            yield frame
                        signal_task = (
                            asyncio.create_task(elicit_signal.wait())
                            if elicit_signal is not None else None
                        )

                    if agent_task in done:
                        event = agent_task.result()
                        if event is _END:
                            agent_task = None
                            continue
                        # Schedule next event read before processing this
                        # one so signal + next-event always race.
                        agent_task = asyncio.create_task(_next_event())

                        if event.type == EventType.LLM_TEXT_DELTA:
                            delta = event.payload.get("delta", "")
                            if delta:
                                saw_delta = True
                                yield sse_data(
                                    make_chat_chunk(
                                        chat_id, model, delta={"content": delta},
                                    )
                                )
                        elif event.type == EventType.ERROR:
                            err = {
                                "error": {
                                    "message": event.payload.get("message", ""),
                                    "type": event.payload.get("kind", "error"),
                                }
                            }
                            yield sse_data(err)
            finally:
                # Cancel any in-flight tasks before exiting (iterator's
                # __aclose__ is implicit on context exit).
                for t in (agent_task, signal_task):
                    if t is not None and not t.done():
                        t.cancel()
                        with contextlib.suppress(
                            asyncio.CancelledError, Exception,
                        ):
                            await t

            # Final drain — covers elicitations that arrived after the
            # last agent event finished but before stream close.
            async for frame in _drain_elicitations():
                yield frame
        except asyncio.CancelledError:
            entry.agent.cancel()
            raise
        except Exception as exc:
            _logger.exception(
                "chat stream failed",
                extra={
                    "event": "chat_stream_failed",
                    "session_id": entry.session.id,
                    "tenant_id": entry.session.tenant_id,
                    "principal": entry.session.principal,
                    "chat_id": chat_id,
                    "model": model,
                },
            )
            yield sse_data({"error": {"message": str(exc), "type": type(exc).__name__}})

        # provider 不支持流式时没有 delta，补发本轮新增的 assistant 文本
        if not saw_delta:
            for msg in entry.session.messages[baseline_msg_count:]:
                if msg.role == Role.ASSISTANT and msg.content:
                    yield sse_data(
                        make_chat_chunk(chat_id, model, delta={"content": msg.content})
                    )

        yield sse_data(make_chat_chunk(chat_id, model, delta={}, finish_reason="stop"))
        yield SSE_DONE_LINE
