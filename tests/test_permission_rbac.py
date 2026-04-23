from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from topsport_agent.server.rbac import (
    RBACPrincipal,
    _default_principal_resolver,
    require_role,
)
from topsport_agent.types.permission import Role


def _app_with_gated_endpoints(principal_resolver) -> FastAPI:
    app = FastAPI()
    app.dependency_overrides[_default_principal_resolver] = principal_resolver

    @app.get("/admin-only")
    def admin_only(_: RBACPrincipal = require_role(Role.ADMIN)):
        return {"ok": True}

    @app.get("/auditor-or-above")
    def auditor(_: RBACPrincipal = require_role(Role.AUDITOR)):
        return {"ok": True}

    return app


def test_admin_endpoint_allows_admin():
    def principal_admin():
        return RBACPrincipal(user_id="alice", tenant_id="acme", role=Role.ADMIN)
    client = TestClient(_app_with_gated_endpoints(principal_admin))
    assert client.get("/admin-only").status_code == 200


def test_admin_endpoint_rejects_operator():
    def principal_op():
        return RBACPrincipal(user_id="bob", tenant_id="acme", role=Role.OPERATOR)
    client = TestClient(_app_with_gated_endpoints(principal_op))
    r = client.get("/admin-only")
    assert r.status_code == 403
    assert "role" in r.json()["detail"].lower()


def test_auditor_endpoint_accepts_admin_and_operator_and_auditor():
    for role in (Role.ADMIN, Role.OPERATOR, Role.AUDITOR):
        def mk(r=role):
            return RBACPrincipal(user_id="x", tenant_id="t", role=r)
        client = TestClient(_app_with_gated_endpoints(mk))
        assert client.get("/auditor-or-above").status_code == 200


def test_auditor_endpoint_rejects_agent():
    def principal_agent():
        return RBACPrincipal(user_id="bot", tenant_id="acme", role=Role.AGENT)
    client = TestClient(_app_with_gated_endpoints(principal_agent))
    assert client.get("/auditor-or-above").status_code == 403
