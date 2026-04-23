from __future__ import annotations

import pytest

from topsport_agent.types.permission import (
    AuditEntry,
    Permission,
    Persona,
    PersonaAssignment,
    Role,
)


def test_permission_enum_values():
    assert Permission.FS_READ == "fs.read"
    assert Permission.SHELL_FULL == "shell.full"
    assert Permission.MCP_GITHUB == "mcp.github"


def test_persona_is_frozen():
    p = Persona(
        id="dev", display_name="Developer", description="d",
        permissions=frozenset({Permission.FS_READ}),
    )
    with pytest.raises(Exception):
        p.id = "other"  # type: ignore[misc]


def test_persona_permissions_is_frozenset():
    p = Persona(
        id="dev", display_name="Developer", description="d",
        permissions=frozenset({Permission.FS_READ, Permission.FS_WRITE}),
    )
    assert isinstance(p.permissions, frozenset)


def test_persona_assignment_defaults():
    a = PersonaAssignment(
        tenant_id="acme",
        persona_ids=frozenset({"dev"}),
        default_persona_id="dev",
    )
    assert a.user_id is None
    assert a.group_id is None


def test_audit_entry_frozen_and_reserved_fields():
    from datetime import datetime, timezone
    e = AuditEntry(
        id="e1", tenant_id="acme", session_id="s",
        user_id=None, persona_id="dev",
        tool_name="read_file",
        tool_required=frozenset({Permission.FS_READ}),
        subject_granted=frozenset({Permission.FS_READ}),
        outcome="allowed",
        args_preview={"path": "/tmp/x"},
        reason=None,
        timestamp=datetime.now(timezone.utc),
    )
    assert e.cost_tokens == 0
    assert e.cost_latency_ms == 0
    assert e.group_id is None
    with pytest.raises(Exception):
        e.outcome = "error"  # type: ignore[misc]


def test_role_enum():
    assert Role.ADMIN == "admin"
    assert Role.OPERATOR == "operator"
    assert Role.AUDITOR == "auditor"
    assert Role.AGENT == "agent"


def test_legacy_symbols_still_importable_but_warn():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from topsport_agent.types.permission import PermissionBehavior  # noqa: F401
        assert any("deprecated" in str(x.message).lower() for x in w)
