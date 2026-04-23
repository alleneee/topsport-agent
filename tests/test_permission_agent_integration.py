from __future__ import annotations

import pytest

from topsport_agent.agent.base import Agent, AgentConfig
from topsport_agent.engine.permission.persona_registry import (
    InMemoryPersonaRegistry,
)
from topsport_agent.types.permission import Permission, Persona


class _FakeProvider:
    name = "fake"
    async def complete(self, request):
        from topsport_agent.llm.response import LLMResponse
        return LLMResponse(text="", tool_calls=[], finish_reason="end_turn",
                           usage={}, response_metadata=None)


@pytest.mark.asyncio
async def test_agent_session_receives_persona_permissions():
    dev = Persona(
        id="dev", display_name="Dev", description="",
        permissions=frozenset({Permission.FS_READ, Permission.SHELL_FULL}),
    )
    registry = InMemoryPersonaRegistry()
    await registry.put(dev)
    agent = Agent.from_config(
        _FakeProvider(),
        AgentConfig(
            model="m",
            persona="dev",
            persona_registry=registry,
            tenant_id="acme",
        ),
    )
    session = await agent.new_session_async()
    assert Permission.FS_READ in session.granted_permissions
    assert Permission.SHELL_FULL in session.granted_permissions
    assert session.persona_id == "dev"
    assert session.tenant_id == "acme"


@pytest.mark.asyncio
async def test_agent_persona_object_accepted_directly():
    dev = Persona(
        id="dev", display_name="Dev", description="",
        permissions=frozenset({Permission.FS_READ}),
    )
    agent = Agent.from_config(
        _FakeProvider(),
        AgentConfig(model="m", persona=dev, tenant_id="acme"),
    )
    session = await agent.new_session_async()
    assert session.persona_id == "dev"
    assert Permission.FS_READ in session.granted_permissions


@pytest.mark.asyncio
async def test_agent_no_persona_means_empty_grants():
    """Back-compat: no persona configured → session has no grants."""
    agent = Agent.from_config(_FakeProvider(), AgentConfig(model="m"))
    session = await agent.new_session_async()
    assert session.granted_permissions == frozenset()
    assert session.persona_id is None
