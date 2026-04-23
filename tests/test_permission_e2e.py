from __future__ import annotations

import pytest

from topsport_agent.agent.base import Agent, AgentConfig
from topsport_agent.engine.permission.audit import (
    AuditLogger,
    InMemoryAuditStore,
)
from topsport_agent.engine.permission.filter import ToolVisibilityFilter
from topsport_agent.engine.permission.redaction import PIIRedactor
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.permission import Permission, Persona


class _P:
    name = "p"

    def __init__(self, rs):
        self._rs, self._i = rs, 0

    async def complete(self, _: LLMRequest) -> LLMResponse:
        r = self._rs[self._i]
        self._i += 1
        return r


@pytest.mark.asyncio
async def test_end_to_end_persona_filters_and_audits():
    dev = Persona(
        id="dev",
        display_name="D",
        description="",
        permissions=frozenset({Permission.FS_READ}),
    )
    audit_store = InMemoryAuditStore()
    audit = AuditLogger(store=audit_store, redactor=PIIRedactor.with_defaults())
    agent = Agent.from_config(
        _P([
            LLMResponse(
                text="done",
                tool_calls=[],
                finish_reason="end_turn",
                usage={},
                response_metadata=None,
            ),
        ]),
        AgentConfig(
            model="m",
            persona=dev,
            tenant_id="acme",
            enable_skills=False,
            enable_memory=False,
            enable_plugins=False,
            permission_filter=ToolVisibilityFilter(audit_logger=audit),
            audit_logger=audit,
        ),
    )
    session = await agent.new_session_async()
    async for _ in agent.engine.run(session):
        pass
    # Session correctly wired
    assert session.persona_id == "dev"
    assert Permission.FS_READ in session.granted_permissions
