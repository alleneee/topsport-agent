"""Admin HTTP API for the permission subsystem.

Exposes CRUD on personas, persona assignments (tenant/user/group → persona
mapping), audit query, and the per-tenant kill-switch. All endpoints are
role-gated via server.rbac.require_role.

Assignment endpoints close the control-plane gap that previously existed: the
execution plane (Engine + ToolVisibilityFilter) looks up `session.granted_permissions`
populated from a Persona, but there was no HTTP path to *bind* a Persona to
a tenant/user/group. Without Assignment, operators had no way to actually
grant capabilities to a running session from outside the code.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..engine.permission.assignment import AssignmentStore
from ..engine.permission.audit import AuditStore
from ..engine.permission.killswitch import KillSwitchGate
from ..engine.permission.persona_registry import PersonaRegistry
from ..types.permission import Persona, PersonaAssignment, Role
from .rbac import require_role

__all__ = ["build_permission_router"]


class _PersonaPayload(BaseModel):
    id: str
    display_name: str
    description: str
    permissions: list[str] = Field(default_factory=list)
    version: int = 1


class _KillSwitchPayload(BaseModel):
    active: bool


class _AssignmentPayload(BaseModel):
    """Tenant/user/group → persona binding.

    Either user_id or group_id (not both) may be set; omit both for a
    tenant-wide default. default_persona_id must be one of persona_ids
    when non-null.
    """

    tenant_id: str
    persona_ids: list[str] = Field(default_factory=list)
    default_persona_id: str | None = None
    user_id: str | None = None
    group_id: str | None = None


def build_permission_router(
    *,
    persona_registry: PersonaRegistry,
    audit_store: AuditStore,
    kill_switch: KillSwitchGate,
    assignment_store: AssignmentStore | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/personas")
    async def list_personas(_=require_role(Role.OPERATOR)):
        ps = await persona_registry.list()
        return [
            {
                "id": p.id,
                "display_name": p.display_name,
                "description": p.description,
                "permissions": sorted(p.permissions),
                "version": p.version,
            }
            for p in ps
        ]

    @router.put("/personas/{persona_id}")
    async def put_persona(
        persona_id: str,
        payload: _PersonaPayload,
        _=require_role(Role.ADMIN),
    ):
        if payload.id != persona_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"payload.id {payload.id!r} != path {persona_id!r}",
            )
        persona = Persona(
            id=payload.id,
            display_name=payload.display_name,
            description=payload.description,
            permissions=frozenset(payload.permissions),
            version=payload.version,
        )
        await persona_registry.put(persona)
        return {"ok": True}

    @router.delete("/personas/{persona_id}")
    async def delete_persona(persona_id: str, _=require_role(Role.ADMIN)):
        await persona_registry.delete(persona_id)
        return {"ok": True}

    @router.post("/killswitch/{tenant_id}")
    async def toggle_killswitch(
        tenant_id: str,
        payload: _KillSwitchPayload,
        _=require_role(Role.ADMIN),
    ):
        if payload.active:
            kill_switch.activate(tenant_id)
        else:
            kill_switch.deactivate(tenant_id)
        return {"active": kill_switch.is_active(tenant_id)}

    @router.get("/killswitch/{tenant_id}")
    async def get_killswitch(tenant_id: str, _=require_role(Role.OPERATOR)):
        return {"active": kill_switch.is_active(tenant_id)}

    # ------------------------------------------------------------------
    # Persona assignments (tenant / user / group → persona binding)
    # ------------------------------------------------------------------

    def _require_assignments() -> AssignmentStore:
        if assignment_store is None:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="assignment_store not configured on this server",
            )
        return assignment_store

    @router.put("/assignments")
    async def put_assignment(
        payload: _AssignmentPayload,
        _=require_role(Role.ADMIN),
    ):
        if payload.user_id and payload.group_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="set at most one of user_id / group_id",
            )
        if (
            payload.default_persona_id is not None
            and payload.default_persona_id not in payload.persona_ids
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="default_persona_id must be in persona_ids",
            )
        store = _require_assignments()
        assignment = PersonaAssignment(
            tenant_id=payload.tenant_id,
            persona_ids=frozenset(payload.persona_ids),
            default_persona_id=payload.default_persona_id,
            user_id=payload.user_id,
            group_id=payload.group_id,
        )
        await store.put(assignment)
        return {"ok": True}

    @router.get("/assignments")
    async def get_assignment(
        tenant_id: str,
        user_id: str | None = None,
        group_id: str | None = None,
        _=require_role(Role.OPERATOR),
    ):
        store = _require_assignments()
        assignment = await store.get(
            tenant_id=tenant_id, user_id=user_id, group_id=group_id,
        )
        if assignment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="assignment not found",
            )
        return {
            "tenant_id": assignment.tenant_id,
            "persona_ids": sorted(assignment.persona_ids),
            "default_persona_id": assignment.default_persona_id,
            "user_id": assignment.user_id,
            "group_id": assignment.group_id,
        }

    @router.delete("/assignments")
    async def delete_assignment(
        tenant_id: str,
        user_id: str | None = None,
        group_id: str | None = None,
        _=require_role(Role.ADMIN),
    ):
        store = _require_assignments()
        await store.delete(
            tenant_id=tenant_id, user_id=user_id, group_id=group_id,
        )
        return {"ok": True}

    @router.get("/audit")
    async def list_audit(
        tenant_id: str,
        limit: int = 100,
        _=require_role(Role.AUDITOR),
    ):
        entries = await audit_store.query(tenant_id=tenant_id, limit=limit)
        return [
            {
                "id": e.id,
                "tenant_id": e.tenant_id,
                "session_id": e.session_id,
                "user_id": e.user_id,
                "persona_id": e.persona_id,
                "tool_name": e.tool_name,
                "tool_required": sorted(e.tool_required),
                "subject_granted": sorted(e.subject_granted),
                "outcome": e.outcome,
                "reason": e.reason,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in entries
        ]

    return router
