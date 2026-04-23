"""Admin HTTP API for the permission subsystem.

Exposes CRUD on personas, assignments (via separate helpers), audit query,
and the per-tenant kill-switch. All endpoints are role-gated via
server.rbac.require_role.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..engine.permission.audit import AuditStore
from ..engine.permission.killswitch import KillSwitchGate
from ..engine.permission.persona_registry import PersonaRegistry
from ..types.permission import Persona, Role
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


def build_permission_router(
    *,
    persona_registry: PersonaRegistry,
    audit_store: AuditStore,
    kill_switch: KillSwitchGate,
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
