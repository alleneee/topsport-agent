from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from topsport_agent.engine.permission.audit import (
    AuditLogger,
    FileAuditStore,
    InMemoryAuditStore,
)
from topsport_agent.engine.permission.redaction import PIIRedactor
from topsport_agent.types.permission import AuditEntry, Permission
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolSpec


async def _noop(args, ctx):
    return "ok"


def _session() -> Session:
    return Session(
        id="s1", system_prompt="",
        tenant_id="acme", principal="alice",
        granted_permissions=frozenset({Permission.FS_READ}),
        persona_id="dev",
    )


def _entry(outcome: str = "allowed") -> AuditEntry:
    return AuditEntry(
        id="e1", tenant_id="acme", session_id="s1",
        user_id="alice", persona_id="dev",
        tool_name="read_file",
        tool_required=frozenset({Permission.FS_READ}),
        subject_granted=frozenset({Permission.FS_READ}),
        outcome=outcome, args_preview={"path": "/tmp/x"},
        reason=None, timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_memory_store_append_and_query():
    store = InMemoryAuditStore()
    await store.append(_entry("allowed"))
    await store.append(_entry("filtered_out"))
    entries = await store.query(tenant_id="acme", limit=10)
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_memory_store_filters_by_tenant():
    store = InMemoryAuditStore()
    await store.append(_entry("allowed"))
    wrong_tenant = _entry("allowed")
    object.__setattr__(wrong_tenant, "tenant_id", "other")
    await store.append(wrong_tenant)
    entries = await store.query(tenant_id="acme", limit=10)
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_file_store_persists_across_instances(tmp_path: Path):
    s1 = FileAuditStore(tmp_path / "audit.jsonl")
    await s1.append(_entry("allowed"))
    s2 = FileAuditStore(tmp_path / "audit.jsonl")
    entries = await s2.query(tenant_id="acme", limit=10)
    assert len(entries) == 1
    assert entries[0].tool_name == "read_file"


@pytest.mark.asyncio
async def test_file_store_is_append_only_jsonl(tmp_path: Path):
    store = FileAuditStore(tmp_path / "audit.jsonl")
    await store.append(_entry())
    await store.append(_entry())
    content = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    lines = [l for l in content.split("\n") if l]
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert parsed["tool_name"] == "read_file"


@pytest.mark.asyncio
async def test_audit_logger_log_call_redacts_args():
    store = InMemoryAuditStore()
    redactor = PIIRedactor.with_defaults()
    logger = AuditLogger(store=store, redactor=redactor)
    session = _session()
    tool = ToolSpec(
        name="read_file", description="", parameters={}, handler=_noop,
        required_permissions=frozenset({Permission.FS_READ}),
    )
    await logger.log_call(
        session=session, tool=tool,
        args={"path": "/home/alice", "contact": "bob@x.com"},
        outcome="allowed", reason=None,
    )
    entries = await store.query(tenant_id="acme", limit=10)
    assert len(entries) == 1
    assert "bob@x.com" not in str(entries[0].args_preview)


def test_audit_logger_log_filtered_is_sync_not_async():
    """ToolVisibilityFilter calls log_filtered from a sync path; verify signature."""
    import inspect
    assert not inspect.iscoroutinefunction(AuditLogger.log_filtered)
    assert not inspect.iscoroutinefunction(AuditLogger.log_killswitch_blocked)
