from __future__ import annotations

from topsport_agent.types.permission import Permission
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


async def _noop(args, ctx):
    return "ok"


def test_toolspec_required_permissions_default_empty():
    spec = ToolSpec(name="t", description="", parameters={}, handler=_noop)
    assert spec.required_permissions == frozenset()


def test_toolspec_required_permissions_can_be_set():
    spec = ToolSpec(
        name="t", description="", parameters={}, handler=_noop,
        required_permissions=frozenset({Permission.FS_READ, Permission.FS_WRITE}),
    )
    assert Permission.FS_READ in spec.required_permissions


def test_session_granted_permissions_default_empty():
    s = Session(id="s", system_prompt="p")
    assert s.granted_permissions == frozenset()
    assert s.persona_id is None


def test_session_granted_permissions_accepts_frozenset():
    s = Session(
        id="s", system_prompt="p",
        granted_permissions=frozenset({Permission.FS_READ}),
        persona_id="dev",
    )
    assert Permission.FS_READ in s.granted_permissions
    assert s.persona_id == "dev"
