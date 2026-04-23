"""AuditStore protocol + in-memory / file implementations + AuditLogger helper.

The logger is the engine-facing API; stores are pluggable backends (memory
and file shipped by default; Postgres/Redis left to the deployment).

File storage is append-only JSON Lines. Each line is one AuditEntry
serialized via json.dumps with datetime isoformatted and frozensets converted
to sorted lists. Load-on-query reads the whole file; acceptable for the
default backend (not production-scale; swap store impl for real volume).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ...types.permission import AuditEntry

if TYPE_CHECKING:
    from ...types.session import Session
    from ...types.tool import ToolSpec
    from .redaction import PIIRedactor

_logger = logging.getLogger(__name__)

__all__ = ["AuditLogger", "AuditStore", "FileAuditStore", "InMemoryAuditStore"]


class AuditStore(Protocol):
    async def append(self, entry: AuditEntry) -> None: ...

    async def query(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[AuditEntry]: ...


class InMemoryAuditStore:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    async def append(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    async def query(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[AuditEntry]:
        result = self._entries
        if tenant_id is not None:
            result = [e for e in result if e.tenant_id == tenant_id]
        if since is not None:
            result = [e for e in result if e.timestamp >= since]
        return result[-limit:]


class FileAuditStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def append(self, entry: AuditEntry) -> None:
        payload = _entry_to_jsonable(entry)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        # Append-only; single-write is atomic on POSIX filesystems for small lines.
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def query(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[AuditEntry]:
        if not self._path.exists():
            return []
        out: list[AuditEntry] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry = _entry_from_jsonable(data)
                if tenant_id is not None and entry.tenant_id != tenant_id:
                    continue
                if since is not None and entry.timestamp < since:
                    continue
                out.append(entry)
        return out[-limit:]


def _entry_to_jsonable(entry: AuditEntry) -> dict[str, Any]:
    d = asdict(entry)
    d["tool_required"] = sorted(entry.tool_required)
    d["subject_granted"] = sorted(entry.subject_granted)
    d["timestamp"] = entry.timestamp.isoformat()
    return d


def _entry_from_jsonable(data: dict[str, Any]) -> AuditEntry:
    return AuditEntry(
        id=data["id"],
        tenant_id=data["tenant_id"],
        session_id=data["session_id"],
        user_id=data.get("user_id"),
        persona_id=data.get("persona_id"),
        tool_name=data["tool_name"],
        tool_required=frozenset(data.get("tool_required", [])),
        subject_granted=frozenset(data.get("subject_granted", [])),
        outcome=data["outcome"],
        args_preview=data.get("args_preview", {}),
        reason=data.get("reason"),
        timestamp=datetime.fromisoformat(data["timestamp"]),
        cost_tokens=data.get("cost_tokens", 0),
        cost_latency_ms=data.get("cost_latency_ms", 0),
        group_id=data.get("group_id"),
    )


class AuditLogger:
    """Engine-facing API: async for per-call audit and filter-time logs.

    log_filtered / log_killswitch_blocked persist one AuditEntry per hidden/blocked
    tool so compliance auditors can query these events via the HTTP API.
    """

    def __init__(
        self,
        *,
        store: AuditStore,
        redactor: "PIIRedactor | None" = None,
    ) -> None:
        self._store = store
        self._redactor = redactor

    async def log_call(
        self,
        *,
        session: "Session",
        tool: "ToolSpec | None",
        args: dict[str, Any],
        outcome: str,
        reason: str | None,
    ) -> None:
        preview = self._redactor.redact_and_truncate(args) if self._redactor else args
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            tenant_id=session.tenant_id or "",
            session_id=session.id,
            user_id=session.principal,
            persona_id=session.persona_id,
            tool_name=tool.name if tool else "<unknown>",
            tool_required=tool.required_permissions if tool else frozenset(),
            subject_granted=session.granted_permissions,
            outcome=outcome,  # type: ignore[arg-type]
            args_preview=preview,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
        )
        try:
            await self._store.append(entry)
        except Exception as exc:
            _logger.error("audit store append failed: %r", exc, exc_info=True)

    async def log_filtered(
        self, session: "Session", filtered_tools: "list[ToolSpec]"
    ) -> None:
        """Persist one AuditEntry per tool hidden by capability filter.

        outcome="filtered_out"；reason=None（过滤是静态策略，无需人类可读原因）。
        args_preview 为空，因为过滤发生在 LLM 生成 tool_call 之前。
        """
        _logger.debug(
            "filtered out %d tools for session %s (tenant=%s, persona=%s): %s",
            len(filtered_tools), session.id, session.tenant_id, session.persona_id,
            [t.name for t in filtered_tools],
        )
        now = datetime.now(timezone.utc)
        for tool in filtered_tools:
            entry = AuditEntry(
                id=str(uuid.uuid4()),
                tenant_id=session.tenant_id or "",
                session_id=session.id,
                user_id=session.principal,
                persona_id=session.persona_id,
                tool_name=tool.name,
                tool_required=tool.required_permissions,
                subject_granted=session.granted_permissions,
                outcome="filtered_out",
                args_preview={},
                reason=None,
                timestamp=now,
            )
            try:
                await self._store.append(entry)
            except Exception as exc:
                _logger.error(
                    "audit store append failed (filtered_out): %r", exc, exc_info=True,
                )

    async def log_killswitch_blocked(
        self, session: "Session", blocked_tools: "list[ToolSpec]"
    ) -> None:
        """Persist one AuditEntry per tool blocked by the tenant kill-switch."""
        _logger.warning(
            "kill-switch blocked %d tools for session %s (tenant=%s)",
            len(blocked_tools), session.id, session.tenant_id,
        )
        now = datetime.now(timezone.utc)
        for tool in blocked_tools:
            entry = AuditEntry(
                id=str(uuid.uuid4()),
                tenant_id=session.tenant_id or "",
                session_id=session.id,
                user_id=session.principal,
                persona_id=session.persona_id,
                tool_name=tool.name,
                tool_required=tool.required_permissions,
                subject_granted=session.granted_permissions,
                outcome="killed",
                args_preview={},
                reason="kill-switch active",
                timestamp=now,
            )
            try:
                await self._store.append(entry)
            except Exception as exc:
                _logger.error(
                    "audit store append failed (killed): %r", exc, exc_info=True,
                )
