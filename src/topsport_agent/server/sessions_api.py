"""/v1/sessions/... 路由：GDPR 导出 / 删除端点。

只暴露 user_hint（客户端传入 `body.user` 的那部分）；服务端自动拼 principal 前缀。
跨 principal 访问因为命名空间隔离无法命中。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import namespace_session_id, require_principal
from .sessions import SessionStore

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/sessions")


def _serialize_messages(messages: list) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        out.append(
            {
                "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in (msg.tool_calls or [])
                ],
                "tool_results": [
                    {
                        "call_id": tr.call_id,
                        "output": tr.output,
                        "is_error": tr.is_error,
                    }
                    for tr in (msg.tool_results or [])
                ],
            }
        )
    return out


@router.get("/{user_hint}")
async def export_session(
    user_hint: str,
    request: Request,
    principal: str = Depends(require_principal),
) -> dict:
    """GDPR 导出：返回完整消息历史 + 元数据。"""
    store: SessionStore = request.app.state.session_store
    sid = namespace_session_id(principal, user_hint)
    entry = await store.get(sid)
    if entry is None:
        raise HTTPException(404, detail=f"session not found: {user_hint!r}")

    session = entry.session
    return {
        "session_id": sid,
        "user_hint": user_hint,
        "principal": principal,
        "state": session.state.value if hasattr(session.state, "value") else str(session.state),
        "system_prompt": session.system_prompt,
        "goal": session.goal,
        "token_spent": session.token_spent,
        "token_budget": session.token_budget,
        "message_count": len(session.messages),
        "messages": _serialize_messages(session.messages),
    }


@router.delete("/{user_hint}", status_code=204)
async def delete_session(
    user_hint: str,
    request: Request,
    principal: str = Depends(require_principal),
) -> None:
    """GDPR 删除：移除 session + 关闭其 agent；404 则说明本就不存在或不归属此 principal。"""
    store: SessionStore = request.app.state.session_store
    sid = namespace_session_id(principal, user_hint)
    deleted = await store.delete(sid)
    if not deleted:
        raise HTTPException(404, detail=f"session not found: {user_hint!r}")
    _logger.info(
        "session deleted: principal=%s user_hint=%s sid=%s",
        principal,
        user_hint,
        sid,
    )


@router.get("")
async def list_sessions(
    request: Request,
    principal: str = Depends(require_principal),
) -> dict:
    """列出当前 principal 拥有的所有 session id（只返回 user_hint 部分）。"""
    store: SessionStore = request.app.state.session_store
    sids = await store.ids_with_prefix(principal + "::")
    # 把 "principal::hint" 还原成 hint 供客户端消费
    hints = [sid.split("::", 1)[1] if "::" in sid else "" for sid in sids]
    return {
        "principal": principal,
        "sessions": hints,
        "count": len(hints),
    }
