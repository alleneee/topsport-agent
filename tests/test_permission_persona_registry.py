from __future__ import annotations

import json
from pathlib import Path

import pytest

from topsport_agent.engine.permission.persona_registry import (
    FilePersonaRegistry,
    InMemoryPersonaRegistry,
)
from topsport_agent.types.permission import Permission, Persona


def _persona(pid: str = "dev") -> Persona:
    return Persona(
        id=pid, display_name=pid.title(), description="test",
        permissions=frozenset({Permission.FS_READ, Permission.FS_WRITE}),
    )


@pytest.mark.asyncio
async def test_memory_registry_put_and_get():
    r = InMemoryPersonaRegistry()
    p = _persona("dev")
    await r.put(p)
    got = await r.get("dev")
    assert got == p


@pytest.mark.asyncio
async def test_memory_registry_get_missing_returns_none():
    r = InMemoryPersonaRegistry()
    assert await r.get("missing") is None


@pytest.mark.asyncio
async def test_memory_registry_list_returns_all():
    r = InMemoryPersonaRegistry()
    await r.put(_persona("dev"))
    await r.put(_persona("ops"))
    ps = await r.list()
    assert len(ps) == 2
    assert {p.id for p in ps} == {"dev", "ops"}


@pytest.mark.asyncio
async def test_memory_registry_put_overwrites():
    r = InMemoryPersonaRegistry()
    await r.put(_persona("dev"))
    new_dev = Persona(
        id="dev", display_name="Dev v2", description="updated",
        permissions=frozenset({Permission.SHELL_FULL}), version=2,
    )
    await r.put(new_dev)
    got = await r.get("dev")
    assert got is not None
    assert got.version == 2
    assert Permission.SHELL_FULL in got.permissions


@pytest.mark.asyncio
async def test_memory_registry_delete():
    r = InMemoryPersonaRegistry()
    await r.put(_persona("dev"))
    await r.delete("dev")
    assert await r.get("dev") is None


@pytest.mark.asyncio
async def test_file_registry_persists(tmp_path: Path):
    path = tmp_path / "personas.json"
    r1 = FilePersonaRegistry(path)
    await r1.put(_persona("dev"))
    r2 = FilePersonaRegistry(path)
    got = await r2.get("dev")
    assert got is not None
    assert Permission.FS_READ in got.permissions


@pytest.mark.asyncio
async def test_file_registry_missing_file_returns_empty(tmp_path: Path):
    r = FilePersonaRegistry(tmp_path / "nope.json")
    assert await r.list() == []
