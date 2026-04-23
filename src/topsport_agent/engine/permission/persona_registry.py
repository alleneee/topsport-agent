"""PersonaRegistry Protocol + in-memory / file backends.

Persona definitions are managed by platform administrators (ADMIN role via
the HTTP admin API). Registry is read by the session-creation path to
resolve Persona.permissions into Session.granted_permissions.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol

from ...types.permission import Permission, Persona

__all__ = [
    "FilePersonaRegistry",
    "InMemoryPersonaRegistry",
    "PersonaRegistry",
]


class PersonaRegistry(Protocol):
    async def get(self, persona_id: str) -> Persona | None: ...
    async def list(self) -> list[Persona]: ...
    async def put(self, persona: Persona) -> None: ...
    async def delete(self, persona_id: str) -> None: ...


class InMemoryPersonaRegistry:
    def __init__(self) -> None:
        self._store: dict[str, Persona] = {}
        self._lock = asyncio.Lock()

    async def get(self, persona_id: str) -> Persona | None:
        async with self._lock:
            return self._store.get(persona_id)

    async def list(self) -> list[Persona]:
        async with self._lock:
            return list(self._store.values())

    async def put(self, persona: Persona) -> None:
        async with self._lock:
            self._store[persona.id] = persona

    async def delete(self, persona_id: str) -> None:
        async with self._lock:
            self._store.pop(persona_id, None)


class FilePersonaRegistry:
    """JSON-file backed. Whole-file read/write per op — single-admin scale only."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def get(self, persona_id: str) -> Persona | None:
        personas = await self._load()
        return personas.get(persona_id)

    async def list(self) -> list[Persona]:
        personas = await self._load()
        return list(personas.values())

    async def put(self, persona: Persona) -> None:
        async with self._lock:
            personas = self._load_sync()
            personas[persona.id] = persona
            self._save_sync(personas)

    async def delete(self, persona_id: str) -> None:
        async with self._lock:
            personas = self._load_sync()
            personas.pop(persona_id, None)
            self._save_sync(personas)

    async def _load(self) -> dict[str, Persona]:
        async with self._lock:
            return self._load_sync()

    def _load_sync(self) -> dict[str, Persona]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        return {
            pid: Persona(
                id=data["id"],
                display_name=data["display_name"],
                description=data["description"],
                permissions=frozenset(
                    Permission(p) if p in Permission._value2member_map_ else p
                    for p in data.get("permissions", [])
                ),
                version=data.get("version", 1),
            )
            for pid, data in raw.items()
        }

    def _save_sync(self, personas: dict[str, Persona]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            pid: {
                "id": p.id,
                "display_name": p.display_name,
                "description": p.description,
                "permissions": sorted(p.permissions),
                "version": p.version,
            }
            for pid, p in personas.items()
        }
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
