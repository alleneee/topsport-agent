from __future__ import annotations

import pytest

from topsport_agent.engine.permission.filter import ToolVisibilityFilter
from topsport_agent.types.permission import Permission
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolSpec


async def _noop(args, ctx):
    return "ok"


def _tool(name: str, perms: frozenset[str] = frozenset()) -> ToolSpec:
    return ToolSpec(
        name=name, description="", parameters={}, handler=_noop,
        required_permissions=perms,
    )


def _session(granted: frozenset[str] = frozenset()) -> Session:
    return Session(id="s", system_prompt="p", granted_permissions=granted)


def test_filter_passes_tools_with_subset_requirements():
    f = ToolVisibilityFilter()
    tools = [
        _tool("read", frozenset({Permission.FS_READ})),
        _tool("write", frozenset({Permission.FS_WRITE})),
    ]
    session = _session(granted=frozenset({Permission.FS_READ}))
    visible = f.filter(tools, session)
    assert [t.name for t in visible] == ["read"]


def test_filter_empty_requirements_always_visible():
    """Tools with no required_permissions are visible regardless of grants."""
    f = ToolVisibilityFilter()
    tools = [_tool("any", frozenset())]
    session = _session(granted=frozenset())
    visible = f.filter(tools, session)
    assert len(visible) == 1


def test_filter_empty_grants_hides_all_guarded_tools():
    f = ToolVisibilityFilter()
    tools = [
        _tool("public", frozenset()),
        _tool("guarded", frozenset({Permission.FS_READ})),
    ]
    session = _session(granted=frozenset())
    visible = f.filter(tools, session)
    assert [t.name for t in visible] == ["public"]


def test_filter_requires_full_subset():
    """If tool needs [A,B] and session has only [A], tool is hidden."""
    f = ToolVisibilityFilter()
    tools = [
        _tool("needs_both", frozenset({Permission.FS_READ, Permission.FS_WRITE})),
    ]
    session = _session(granted=frozenset({Permission.FS_READ}))
    visible = f.filter(tools, session)
    assert visible == []


def test_filter_superset_grants_work():
    f = ToolVisibilityFilter()
    tools = [_tool("read", frozenset({Permission.FS_READ}))]
    session = _session(granted=frozenset({
        Permission.FS_READ, Permission.FS_WRITE, Permission.SHELL_FULL
    }))
    visible = f.filter(tools, session)
    assert len(visible) == 1
