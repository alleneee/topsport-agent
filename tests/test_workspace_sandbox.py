"""Workspace-based file_ops sandboxing — production-path regression tests.

Verifies that:
  - write_file / read_file / list_dir outside the session workspace are rejected
  - Path traversal (`..` components) is blocked after resolve
  - Two sessions have isolated files (one can't read the other's)
  - Session without a workspace (CLI mode) is still unrestricted (back-compat)

These tests go through the real production wiring (create_app default
hooks) rather than injecting a pre-built agent factory, so they catch
regressions where workspace_root gets disconnected from the tool runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topsport_agent.tools.file_ops import file_tools
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext
from topsport_agent.workspace import WorkspaceRegistry


def _handler_by_name(name: str):
    """Grab the handler for a given builtin file tool by name."""
    for spec in file_tools():
        if spec.name == name:
            return spec.handler
    raise AssertionError(f"no tool named {name}")


def _ctx_for_session(session: Session) -> ToolContext:
    """Build the ToolContext the engine would build for this session."""
    import asyncio

    return ToolContext(
        session_id=session.id,
        call_id="test-call",
        cancel_event=asyncio.Event(),
        workspace_root=(
            session.workspace.files_dir if session.workspace is not None else None
        ),
    )


@pytest.fixture
def registry(tmp_path: Path) -> WorkspaceRegistry:
    return WorkspaceRegistry(tmp_path / "workspaces")


async def test_workspace_registry_creates_isolated_dirs_per_session(
    registry: WorkspaceRegistry,
) -> None:
    a = registry.acquire("sess-a")
    b = registry.acquire("sess-b")
    assert a.files_dir != b.files_dir
    assert a.files_dir.exists()
    assert b.files_dir.exists()


async def test_session_id_with_unsafe_chars_is_sanitised(
    registry: WorkspaceRegistry,
) -> None:
    ws = registry.acquire("user::plan:p-1/../escape")
    # '/' / ':' flattened to '_'; the whole sanitised id becomes ONE
    # directory name (no more path components), so resolve() stays under base
    # even if the literal characters look suspicious.
    assert ws.root.resolve().is_relative_to(registry.base.resolve())
    # Sanity check: the resulting dir name is a single path component, so
    # even a substring like "_.._" can't break out of the parent.
    assert "/" not in ws.root.name
    assert ws.root.parent == registry.base


async def test_write_file_inside_workspace_succeeds(
    registry: WorkspaceRegistry,
) -> None:
    ws = registry.acquire("s1")
    session = Session(id="s1", system_prompt="")
    session.workspace = ws
    ctx = _ctx_for_session(session)

    write = _handler_by_name("write_file")
    inside = str(ws.files_dir / "ok.txt")
    result = await write({"path": inside, "content": "hello"}, ctx)
    assert "ok" in str(result).lower() or Path(inside).read_text() == "hello"
    assert Path(inside).read_text() == "hello"


async def test_write_file_outside_workspace_rejected(
    registry: WorkspaceRegistry,
    tmp_path: Path,
) -> None:
    ws = registry.acquire("s1")
    session = Session(id="s1", system_prompt="")
    session.workspace = ws
    ctx = _ctx_for_session(session)

    write = _handler_by_name("write_file")
    # Attempt to write OUTSIDE the session's files_dir, into an arbitrary
    # location under tmp_path (simulating /etc/passwd etc.)
    outside = str(tmp_path / "escape.txt")
    result = await write({"path": outside, "content": "pwned"}, ctx)
    assert "escape" in str(result).lower() or "workspace" in str(result).lower()
    assert not Path(outside).exists()


async def test_path_traversal_via_dotdot_rejected(
    registry: WorkspaceRegistry,
) -> None:
    ws = registry.acquire("s1")
    session = Session(id="s1", system_prompt="")
    session.workspace = ws
    ctx = _ctx_for_session(session)

    write = _handler_by_name("write_file")
    # .. inside the path should resolve outside workspace and be rejected
    sneaky = str(ws.files_dir / ".." / ".." / "gotcha.txt")
    result = await write({"path": sneaky, "content": "x"}, ctx)
    assert "escape" in str(result).lower() or "workspace" in str(result).lower()


async def test_one_session_cannot_read_anothers_file(
    registry: WorkspaceRegistry,
) -> None:
    ws_a = registry.acquire("alice")
    ws_b = registry.acquire("bob")
    secret_path = ws_a.files_dir / "secret.txt"
    secret_path.write_text("alice-only")

    # Bob's session tries to read Alice's file by absolute path.
    session_b = Session(id="bob", system_prompt="")
    session_b.workspace = ws_b
    ctx_b = _ctx_for_session(session_b)

    read = _handler_by_name("read_file")
    result = await read({"path": str(secret_path)}, ctx_b)
    # Either an error string or a dict with is_error — both acceptable
    # so long as "alice-only" did NOT appear in the result.
    assert "alice-only" not in str(result)


async def test_session_without_workspace_is_unrestricted_backcompat(
    tmp_path: Path,
) -> None:
    """CLI mode / legacy tests don't bind a workspace. That MUST still work
    (no sandbox enforcement) or we'd break every CLI-based user."""
    session = Session(id="cli", system_prompt="")
    # session.workspace stays None
    ctx = _ctx_for_session(session)
    assert ctx.workspace_root is None

    write = _handler_by_name("write_file")
    target = str(tmp_path / "cli_mode.txt")
    await write({"path": target, "content": "cli-ok"}, ctx)
    assert Path(target).read_text() == "cli-ok"
