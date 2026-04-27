"""Server-side elicitation broker: route MCP server-initiated user-input
requests through the existing chat SSE stream and accept user replies
via a dedicated POST endpoint.

Threading model:
    - Each elicitation request is keyed by a UUID generated at the
      `MCPClient._elicitation_callback` boundary.
    - The broker stashes a `Pending` (asyncio.Future + session affinity
      + timeout) under that UUID.
    - The chat SSE stream (in server/chat.py) polls the broker for the
      session's pending requests on every iteration of its event loop
      and emits `event: elicitation` frames for each.
    - `POST /v1/elicitations/<id>` resolves the future when the user
      answers; broker returns the response to the awaiting MCP handler.
    - Timeout (default 60s, configurable) auto-resolves with a
      `decline` action so MCP servers don't hang indefinitely on a
      user who walked away.

Why this design (vs WebSocket / dedicated SSE channel):
    - Reuses the chat SSE stream that's already authenticated and
      lifetime-bound to the session — no new auth surface.
    - HTTP POST for answers is request/response (idempotent), trivially
      retryable, no connection upgrade.
    - The broker is a single in-memory state container; for multi-replica
      deployments this needs Redis-backed storage (follow-up; not in
      scope for v1).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..mcp.elicitation import (
    ElicitationHandler,
    ElicitationRequest,
    ElicitationResponse,
)

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Pending:
    """An in-flight elicitation: waiting for the user to answer.

    `request`: the question being asked (immutable).
    `session_id`: which user session the question was routed to (used
        by chat SSE stream to filter pending requests).
    `future`: resolved by `POST /v1/elicitations/<id>` with the
        response, or auto-resolved with decline on timeout.
    `delivered`: set when the SSE stream has emitted this request to
        the user; prevents duplicate sends.
    """

    request: ElicitationRequest
    session_id: str
    future: asyncio.Future[ElicitationResponse]
    delivered: bool = field(default=False)


class HTTPElicitationBroker:
    """Routes MCP server-initiated elicitation requests through the
    chat SSE stream and back via POST.

    One singleton per server process, mounted on `app.state.elicitation_broker`.

    Public surface (the elicitation handler / chat stream / POST endpoint
    interact with):
        - `await handle(request)`: called by `MCPClient` adapter; blocks
          until user answers (or timeout). Returns `ElicitationResponse`.
        - `pending_for_session(session_id)`: chat SSE stream calls this
          to discover newly-arrived elicitations for the active session.
          Returns list of `(id, request)` for not-yet-delivered ones,
          and marks them delivered.
        - `resolve(elicitation_id, response)`: POST endpoint hands the
          user's reply here.
    """

    def __init__(self, *, default_timeout_seconds: float = 60.0) -> None:
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be > 0")
        self._timeout = default_timeout_seconds
        self._pending: dict[str, _Pending] = {}
        # Per-session pointer list (id) so chat stream can iterate without
        # walking the full broker dict each cycle.
        self._by_session: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        # Per-session asyncio.Event: set whenever a new elicitation arrives
        # for that session, so chat SSE streams can `asyncio.wait` on it
        # alongside the agent event iterator and respond immediately —
        # critical because LLM thinking time (no agent events emitted)
        # would otherwise let elicitations sit in the queue past the
        # broker's timeout. Lazily allocated; `signal_for` doubles as
        # "subscribe" + getter.
        self._session_signals: dict[str, asyncio.Event] = {}

    @property
    def default_timeout(self) -> float:
        return self._timeout

    async def handle(self, request: ElicitationRequest) -> ElicitationResponse:
        """Block until user answers (or timeout).

        Routing: `request.session_id` is set by
        `MCPClient._elicitation_callback` from the client instance's
        `_current_call_session_id` field (populated by `MCPToolSource`
        under `_call_lock`). ContextVar doesn't work across the SDK
        task boundary; this is the production path.

        If `session_id is None` (request landed outside a tool call —
        e.g. server-initiated background work — or the operator hasn't
        wired tool_bridge), auto-cancel: the user has no UI surface
        to answer through. We use `cancel` (not `decline`) so the
        server knows it's a "no answer" rather than an explicit refusal.
        """
        sid = request.session_id
        if sid is None:
            _logger.warning(
                "elicitation request id=%s arrived without session_id; "
                "auto-cancelling (no UI surface to route through)",
                request.id,
            )
            return ElicitationResponse(action="cancel")

        pending = _Pending(
            request=request, session_id=sid,
            future=asyncio.get_event_loop().create_future(),
        )
        async with self._lock:
            self._pending[request.id] = pending
            self._by_session.setdefault(sid, []).append(request.id)
            # Wake up any chat SSE stream awaiting this session's signal
            # so the new elicitation frame surfaces immediately rather
            # than waiting on the next agent event.
            sig = self._session_signals.get(sid)
            if sig is not None:
                sig.set()

        try:
            return await asyncio.wait_for(
                pending.future, timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            # spec: cancel = no-answer (retryable) vs decline = refused
            # (not retryable). Timeout is "user didn't answer" → cancel.
            _logger.info(
                "elicitation id=%s timed out after %.1fs; auto-cancel",
                request.id, self._timeout,
            )
            return ElicitationResponse(action="cancel")
        finally:
            async with self._lock:
                self._pending.pop(request.id, None)
                ids = self._by_session.get(sid)
                if ids is not None:
                    try:
                        ids.remove(request.id)
                    except ValueError:
                        pass
                    if not ids:
                        self._by_session.pop(sid, None)

    def signal_for(self, session_id: str) -> asyncio.Event:
        """Subscribe a session to a wakeup `asyncio.Event` set whenever
        a new elicitation arrives for that session.

        Chat SSE stream pattern:
            sig = broker.signal_for(sid)
            ...
            done, _ = await asyncio.wait([agent_iter_task, sig.wait_task],
                                          return_when=FIRST_COMPLETED)
            if signal task done:
                sig.clear()
                yield from broker.pending_for_session(sid)
            ...

        The Event is lazily created and shared across subscribers (single
        listener per session is the typical pattern; multiple listeners
        share the same wakeup which is fine since `pending_for_session`
        marks delivered to prevent duplicate sends)."""
        return self._session_signals.setdefault(session_id, asyncio.Event())

    async def pending_for_session(
        self, session_id: str,
    ) -> list[tuple[str, ElicitationRequest]]:
        """Return (id, request) pairs not yet delivered to this session's
        SSE stream. Marks them delivered before returning so the same
        request isn't sent twice."""
        async with self._lock:
            ids = list(self._by_session.get(session_id) or [])
            results: list[tuple[str, ElicitationRequest]] = []
            for rid in ids:
                p = self._pending.get(rid)
                if p is None or p.delivered:
                    continue
                p.delivered = True
                results.append((rid, p.request))
            return results

    async def resolve(
        self,
        elicitation_id: str,
        response: ElicitationResponse,
        *,
        expected_session_id: str | None = None,
    ) -> bool:
        """Resolve the pending elicitation.

        Returns True iff the id was found, not yet resolved, and (if
        `expected_session_id` is provided) the pending was issued for
        that session. Returns False otherwise — the endpoint translates
        False into 404 to avoid leaking whether an id exists for some
        other session (cross-tenant id-probing defence).
        """
        async with self._lock:
            p = self._pending.get(elicitation_id)
            if p is None or p.future.done():
                return False
            if (expected_session_id is not None
                    and p.session_id != expected_session_id):
                # Caller authenticated as a session that doesn't own this
                # elicitation. Don't reveal that the id exists; treat
                # exactly like "not found".
                return False
            p.future.set_result(response)
            return True

    def make_handler(self) -> "ElicitationHandler":
        """Return an `ElicitationHandler` Callable bound to this broker.

        Type-correct: `ElicitationHandler = Callable[[ElicitationRequest],
        Awaitable[ElicitationResponse]]`, which matches the inner async
        function below."""
        async def _handler(request: ElicitationRequest) -> ElicitationResponse:
            return await self.handle(request)

        return _handler


__all__ = [
    "HTTPElicitationBroker",
]
