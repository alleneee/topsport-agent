from __future__ import annotations

import pytest

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.engine.permission.audit import (
    AuditLogger, InMemoryAuditStore,
)
from topsport_agent.engine.permission.filter import ToolVisibilityFilter
from topsport_agent.engine.permission.killswitch import KillSwitchGate
from topsport_agent.engine.permission.redaction import PIIRedactor
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.permission import Permission
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


class _Provider:
    name = "p"
    def __init__(self, rs): self._rs, self._i = list(rs), 0
    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        r = self._rs[self._i]; self._i += 1
        return r


async def _h(args, ctx): return "ok"


def _session(granted: frozenset[str]) -> Session:
    s = Session(
        id="s", system_prompt="p", tenant_id="acme", principal="alice",
        granted_permissions=granted, persona_id="dev",
    )
    s.messages.append(Message(role=Role.USER, content="go"))
    return s


def _filterable_pool() -> list[ToolSpec]:
    return [
        ToolSpec(name="read", description="", parameters={}, handler=_h,
                 required_permissions=frozenset({Permission.FS_READ})),
        ToolSpec(name="write", description="", parameters={}, handler=_h,
                 required_permissions=frozenset({Permission.FS_WRITE})),
    ]


@pytest.mark.asyncio
async def test_engine_hides_tools_outside_grants():
    audit = AuditLogger(store=InMemoryAuditStore(), redactor=None)
    engine = Engine(
        _Provider([
            LLMResponse(text="noop", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]),
        _filterable_pool(),
        EngineConfig(model="m"),
        permission_filter=ToolVisibilityFilter(audit_logger=audit),
        audit_logger=audit,
    )
    session = _session(granted=frozenset({Permission.FS_READ}))
    tools_in_pool: list[ToolSpec] = []
    orig = engine._snapshot_tools
    async def spy():
        r = await orig(session)
        tools_in_pool.extend(r)
        return r
    engine._snapshot_tools = spy  # type: ignore[assignment]
    async for _ in engine.run(session):
        pass
    assert {t.name for t in tools_in_pool} == {"read"}


@pytest.mark.asyncio
async def test_engine_audits_each_tool_call():
    store = InMemoryAuditStore()
    audit = AuditLogger(store=store, redactor=PIIRedactor.with_defaults())
    tool = ToolSpec(
        name="read", description="", parameters={}, handler=_h,
        required_permissions=frozenset({Permission.FS_READ}),
    )
    engine = Engine(
        _Provider([
            LLMResponse(
                text="", tool_calls=[ToolCall(id="c1", name="read",
                                              arguments={"contact": "bob@x.com"})],
                finish_reason="tool_use", usage={}, response_metadata=None,
            ),
            LLMResponse(text="done", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]),
        [tool],
        EngineConfig(model="m"),
        permission_filter=ToolVisibilityFilter(audit_logger=audit),
        audit_logger=audit,
    )
    session = _session(granted=frozenset({Permission.FS_READ}))
    async for _ in engine.run(session):
        pass
    entries = await store.query(tenant_id="acme", limit=10)
    assert len(entries) >= 1
    call_entries = [e for e in entries if e.outcome == "allowed"]
    assert len(call_entries) == 1
    assert "bob@x.com" not in str(call_entries[0].args_preview)


@pytest.mark.asyncio
async def test_killswitch_empties_tool_pool():
    kill = KillSwitchGate()
    kill.activate("acme")
    engine = Engine(
        _Provider([
            LLMResponse(text="noop", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]),
        _filterable_pool(),
        EngineConfig(model="m"),
        permission_filter=ToolVisibilityFilter(kill_switch=kill),
    )
    session = _session(granted=frozenset({
        Permission.FS_READ, Permission.FS_WRITE,
    }))
    seen: list[ToolSpec] = []
    orig = engine._snapshot_tools
    async def spy():
        r = await orig(session)
        seen.extend(r)
        return r
    engine._snapshot_tools = spy  # type: ignore[assignment]
    async for _ in engine.run(session):
        pass
    assert seen == []


@pytest.mark.asyncio
async def test_no_filter_means_no_permission_enforcement():
    """Back-compat: Engine without permission_filter passes all tools through."""
    engine = Engine(
        _Provider([
            LLMResponse(text="noop", tool_calls=[], finish_reason="end_turn",
                        usage={}, response_metadata=None),
        ]),
        _filterable_pool(),
        EngineConfig(model="m"),
    )
    session = _session(granted=frozenset())
    seen: list[ToolSpec] = []
    orig = engine._snapshot_tools
    async def spy():
        r = await orig(session)
        seen.extend(r)
        return r
    engine._snapshot_tools = spy  # type: ignore[assignment]
    async for _ in engine.run(session):
        pass
    assert {t.name for t in seen} == {"read", "write"}
