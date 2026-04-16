from __future__ import annotations

from .registry import SkillRegistry


class SkillMatcher:
    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry
        self._activations: dict[str, list[str]] = {}

    def activate(self, session_id: str, name: str) -> bool:
        if self._registry.get(name) is None:
            return False
        active = self._activations.setdefault(session_id, [])
        if name not in active:
            active.append(name)
        return True

    def deactivate(self, session_id: str, name: str) -> bool:
        active = self._activations.get(session_id)
        if not active or name not in active:
            return False
        active.remove(name)
        return True

    def active_skills(self, session_id: str) -> list[str]:
        return list(self._activations.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._activations.pop(session_id, None)
