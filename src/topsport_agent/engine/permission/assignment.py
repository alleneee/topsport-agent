"""PersonaAssignmentStore + resolution logic.

Resolution precedence:
1. (tenant_id, user_id) — per-user override wins
2. (tenant_id, group_id) — group default
3. (tenant_id, None, None) — tenant-wide fallback
4. None — caller must deny session creation (fail-closed)
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from ...types.permission import PersonaAssignment

__all__ = [
    "AssignmentStore",
    "InMemoryAssignmentStore",
    "resolve_persona_ids",
]


class AssignmentStore(Protocol):
    async def get(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        group_id: str | None = None,
    ) -> PersonaAssignment | None: ...

    async def put(self, assignment: PersonaAssignment) -> None: ...

    async def delete(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        group_id: str | None = None,
    ) -> None: ...


class InMemoryAssignmentStore:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str | None, str | None], PersonaAssignment] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(
        tenant_id: str, user_id: str | None, group_id: str | None
    ) -> tuple[str, str | None, str | None]:
        return (tenant_id, user_id, group_id)

    async def get(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        group_id: str | None = None,
    ) -> PersonaAssignment | None:
        async with self._lock:
            return self._store.get(self._key(tenant_id, user_id, group_id))

    async def put(self, assignment: PersonaAssignment) -> None:
        async with self._lock:
            self._store[self._key(
                assignment.tenant_id, assignment.user_id, assignment.group_id,
            )] = assignment

    async def delete(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        group_id: str | None = None,
    ) -> None:
        async with self._lock:
            self._store.pop(self._key(tenant_id, user_id, group_id), None)


async def resolve_persona_ids(
    store: AssignmentStore,
    *,
    tenant_id: str,
    user_id: str | None = None,
    group_id: str | None = None,
) -> tuple[frozenset[str], str | None] | None:
    """Returns (persona_ids, default_persona_id) or None if no assignment."""
    # Level 1: user-specific
    if user_id is not None:
        a = await store.get(tenant_id=tenant_id, user_id=user_id)
        if a is not None:
            return a.persona_ids, a.default_persona_id
    # Level 2: group
    if group_id is not None:
        a = await store.get(tenant_id=tenant_id, group_id=group_id)
        if a is not None:
            return a.persona_ids, a.default_persona_id
    # Level 3: tenant-wide
    a = await store.get(tenant_id=tenant_id)
    if a is not None:
        return a.persona_ids, a.default_persona_id
    return None
