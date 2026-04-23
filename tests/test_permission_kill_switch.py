from __future__ import annotations

import pytest

from topsport_agent.engine.permission.killswitch import KillSwitchGate


def test_killswitch_default_inactive():
    g = KillSwitchGate()
    assert g.is_active("acme") is False


def test_killswitch_activate_and_deactivate():
    g = KillSwitchGate()
    g.activate("acme")
    assert g.is_active("acme") is True
    assert g.is_active("other") is False
    g.deactivate("acme")
    assert g.is_active("acme") is False


def test_killswitch_none_tenant_safe():
    """Session.tenant_id may be None; must not crash."""
    g = KillSwitchGate()
    assert g.is_active(None) is False


def test_killswitch_active_tenants_snapshot_is_immutable():
    g = KillSwitchGate()
    g.activate("t1")
    g.activate("t2")
    snap = g.active_tenants()
    assert snap == frozenset({"t1", "t2"})
    snap2 = frozenset(snap | {"t3"})
    assert g.active_tenants() == frozenset({"t1", "t2"})
    assert snap2 == frozenset({"t1", "t2", "t3"})
