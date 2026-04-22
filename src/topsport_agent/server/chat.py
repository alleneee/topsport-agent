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
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..types.events import EventType
from ..types.message import Role
from .auth import namespace_session_id, require_principal
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from .sessions import SessionEntry, SessionStore
from .sse import SSE_DONE_LINE, make_chat_chunk, sse_data

_logger = logging.getLogger(__name__)

router = APIRouter()


def _extract_last_user_message(messages: list) -> str | None:
    for m in reversed(messages):
        if m.role == "user" and m.content:
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

    user_input = _extract_last_user_message(body.messages)
    if not user_input:
        raise HTTPException(400, detail="messages must contain at least one user message")

    # principal 前缀隔离：不同 principal 即便传同一个 body.user 也命中不同 session
    session_key = namespace_session_id(principal, body.user)
    # tenant = principal（最简映射）：sandbox / per-tenant quota / 审计都按 principal 维度做。
    _, entry, _ = await store.get_or_create(
        session_key, model_name, tenant_id=principal, principal=principal,
    )

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.stream:
        return StreamingResponse(
            _stream_chat(entry, user_input, chat_id, body.model, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(await _complete_chat(entry, user_input, chat_id, body.model))


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

        try:
            async for event in entry.agent.run(user_input, entry.session):
                if await request.is_disconnected():
                    entry.agent.cancel()
                    break
                if event.type == EventType.LLM_TEXT_DELTA:
                    delta = event.payload.get("delta", "")
                    if delta:
                        saw_delta = True
                        yield sse_data(
                            make_chat_chunk(chat_id, model, delta={"content": delta})
                        )
                elif event.type == EventType.ERROR:
                    err = {
                        "error": {
                            "message": event.payload.get("message", ""),
                            "type": event.payload.get("kind", "error"),
                        }
                    }
                    yield sse_data(err)
        except asyncio.CancelledError:
            entry.agent.cancel()
            raise
        except Exception as exc:
            _logger.exception("chat stream failed")
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
