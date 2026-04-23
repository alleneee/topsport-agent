from __future__ import annotations

import pytest

from topsport_agent.engine.permission.assignment import (
    InMemoryAssignmentStore,
    resolve_persona_ids,
)
from topsport_agent.types.permission import PersonaAssignment


@pytest.mark.asyncio
async def test_user_specific_assignment_wins():
    store = InMemoryAssignmentStore()
    await store.put(PersonaAssignment(
        tenant_id="acme", user_id="alice",
        persona_ids=frozenset({"dev"}), default_persona_id="dev",
    ))
    await store.put(PersonaAssignment(
        tenant_id="acme",
        persona_ids=frozenset({"viewer"}), default_persona_id="viewer",
    ))
    result = await resolve_persona_ids(store, tenant_id="acme", user_id="alice")
    assert result == (frozenset({"dev"}), "dev")


@pytest.mark.asyncio
async def test_group_fallback_when_no_user_match():
    store = InMemoryAssignmentStore()
    await store.put(PersonaAssignment(
        tenant_id="acme", group_id="eng",
        persona_ids=frozenset({"dev"}), default_persona_id="dev",
    ))
    result = await resolve_persona_ids(
        store, tenant_id="acme", user_id="bob", group_id="eng",
    )
    assert result == (frozenset({"dev"}), "dev")


@pytest.mark.asyncio
async def test_tenant_fallback_when_no_group_match():
    store = InMemoryAssignmentStore()
    await store.put(PersonaAssignment(
        tenant_id="acme",
        persona_ids=frozenset({"viewer"}), default_persona_id="viewer",
    ))
    result = await resolve_persona_ids(store, tenant_id="acme", user_id="bob")
    assert result == (frozenset({"viewer"}), "viewer")


@pytest.mark.asyncio
async def test_no_assignment_returns_none():
    store = InMemoryAssignmentStore()
    result = await resolve_persona_ids(store, tenant_id="acme", user_id="bob")
    assert result is None
