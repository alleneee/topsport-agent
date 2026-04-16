from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SkillManifest:
    name: str
    description: str
    path: Path
    body_path: Path
    extra: dict[str, str] = field(default_factory=dict)
    resources: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class LoadedSkill:
    manifest: SkillManifest
    body: str
    resources_index: list[str] = field(default_factory=list)
