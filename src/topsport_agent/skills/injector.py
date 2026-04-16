from __future__ import annotations

from ..types.message import Message, Role
from ..types.session import Session
from .loader import SkillLoader
from .matcher import SkillMatcher
from .registry import SkillRegistry


class SkillInjector:
    name = "skill"

    def __init__(
        self,
        registry: SkillRegistry,
        loader: SkillLoader,
        matcher: SkillMatcher,
        *,
        catalog_header: str = "Available skills",
        include_catalog: bool = True,
    ) -> None:
        self._registry = registry
        self._loader = loader
        self._matcher = matcher
        self._catalog_header = catalog_header
        self._include_catalog = include_catalog

    async def provide(self, session: Session) -> list[Message]:
        messages: list[Message] = []

        if self._include_catalog:
            catalog = self._render_catalog()
            if catalog:
                messages.append(Message(role=Role.SYSTEM, content=catalog))

        for name in self._matcher.active_skills(session.id):
            loaded = self._loader.load(name)
            if loaded is None:
                continue
            body = (
                f"## Skill: {loaded.manifest.name}\n\n"
                f"{loaded.manifest.description}\n\n"
                f"{loaded.body}"
            )
            messages.append(Message(role=Role.SYSTEM, content=body))

        return messages

    def _render_catalog(self) -> str:
        manifests = self._registry.list()
        if not manifests:
            return ""
        lines = [f"## {self._catalog_header}", ""]
        for manifest in manifests:
            lines.append(f"- `{manifest.name}`: {manifest.description}")
        lines.append("")
        lines.append(
            "Use `load_skill(name=...)` to activate a skill's full instructions for this session."
        )
        return "\n".join(lines)
