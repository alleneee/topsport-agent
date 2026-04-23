from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from topsport_agent.engine.permission.audit import InMemoryAuditStore
from topsport_agent.engine.permission.killswitch import KillSwitchGate
from topsport_agent.engine.permission.persona_registry import (
    InMemoryPersonaRegistry,
)
from topsport_agent.server.permission_api import build_permission_router
from topsport_agent.server.rbac import RBACPrincipal, _default_principal_resolver
from topsport_agent.types.permission import Role


def _client(role: Role):
    registry = InMemoryPersonaRegistry()
    audit = InMemoryAuditStore()
    kill = KillSwitchGate()
    app = FastAPI()
    router = build_permission_router(
        persona_registry=registry,
        audit_store=audit,
        kill_switch=kill,
    )
    app.include_router(router, prefix="/v1/admin")
    def principal():
        return RBACPrincipal(user_id="u", tenant_id="acme", role=role)
    app.dependency_overrides[_default_principal_resolver] = principal
    return TestClient(app), registry, audit, kill


def test_list_personas_requires_operator_or_above():
    client_agent, *_ = _client(Role.AGENT)
    assert client_agent.get("/v1/admin/personas").status_code == 403
    client_op, *_ = _client(Role.OPERATOR)
    assert client_op.get("/v1/admin/personas").status_code == 200


def test_put_persona_admin_only():
    client, registry, _, _ = _client(Role.OPERATOR)
    payload = {
        "id": "dev", "display_name": "Dev", "description": "d",
        "permissions": ["fs.read"], "version": 1,
    }
    assert client.put("/v1/admin/personas/dev", json=payload).status_code == 403

    client, registry, _, _ = _client(Role.ADMIN)
    r = client.put("/v1/admin/personas/dev", json=payload)
    assert r.status_code == 200


def test_killswitch_toggle():
    client, _, _, kill = _client(Role.ADMIN)
    assert kill.is_active("acme") is False
    r = client.post("/v1/admin/killswitch/acme", json={"active": True})
    assert r.status_code == 200
    assert kill.is_active("acme") is True
    r = client.post("/v1/admin/killswitch/acme", json={"active": False})
    assert kill.is_active("acme") is False


def test_audit_query_requires_auditor():
    client, _, _, _ = _client(Role.AGENT)
    assert client.get("/v1/admin/audit?tenant_id=acme").status_code == 403
    client, _, _, _ = _client(Role.AUDITOR)
    r = client.get("/v1/admin/audit?tenant_id=acme")
    assert r.status_code == 200
    assert r.json() == []
