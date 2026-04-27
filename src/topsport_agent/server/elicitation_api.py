"""POST /v1/elicitations/<id> endpoint — accept user replies to MCP
server-initiated elicitation requests routed through the chat SSE stream.

Body:
    {
        "action": "accept" | "decline" | "cancel",
        "content": {...optional form values...}
    }

Response:
    200 if the elicitation was found and resolved.
    404 if the id is unknown (already resolved / timed out / never existed).
    400 if the body is malformed.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from ..mcp.elicitation import ElicitationResponse
from .auth import require_principal

router = APIRouter(prefix="/v1/elicitations")


class ElicitationReplyBody(BaseModel):
    action: Literal["accept", "decline", "cancel"]
    content: dict[str, Any] | None = None


@router.post("/{elicitation_id}")
async def reply_elicitation(
    elicitation_id: str,
    body: ElicitationReplyBody,
    request: Request,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    principal: str = Depends(require_principal),
) -> dict[str, Any]:
    """Resolve a pending elicitation.

    Auth model:
      - Principal must be authenticated (Bearer token).
      - Caller MUST provide `X-Session-Id` header naming the session
        the elicitation was issued for.
      - Broker verifies the elicitation was created for that exact
        session_id; mismatch → 404 (NOT 403, to avoid id-probing
        oracles that leak whether ids exist for other sessions).

    Without `X-Session-Id`, any authenticated user could answer any
    server's elicitation (cross-tenant escape). The header is required
    when the broker has multiple pending requests across sessions.
    """
    del principal  # presence acknowledged via Depends; routing is via X-Session-Id
    broker = getattr(request.app.state, "elicitation_broker", None)
    if broker is None:
        raise HTTPException(
            status_code=404,
            detail="elicitation broker not configured on this server",
        )
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="X-Session-Id header required for elicitation reply",
        )
    response = ElicitationResponse(
        action=body.action, content=body.content,
    )
    ok = await broker.resolve(
        elicitation_id, response, expected_session_id=x_session_id,
    )
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=(
                f"elicitation {elicitation_id!r} not found for this session "
                f"(already resolved / timed out / never existed / "
                f"belongs to a different session)"
            ),
        )
    return {"status": "ok"}


__all__ = ["router"]
