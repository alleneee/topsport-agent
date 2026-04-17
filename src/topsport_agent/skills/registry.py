from __future__ import annotations

from pathlib import Path

from ._frontmatter import parse as parse_frontmatter
from .types import SkillManifest

_RESERVED_KEYS = {"name", "description"}


class SkillRegistry:
    """注册表负责扫描目录树、解析 SKILL.md frontmatter，构建 name -> manifest 映射。

    同名冲突时先扫描到的优先（setdefault），目录顺序决定优先级。
    """
    def __init__(self, skill_dirs: list[Path]) -> None:
        self._skill_dirs = [Path(p) for p in skill_dirs]
        self._manifests: dict[str, SkillManifest] = {}

    def load(self) -> None:
        self._manifests = {}
        for base in self._skill_dirs:
            if not base.exists():
                continue
            for skill_md in sorted(base.rglob("SKILL.md")):
                manifest = self._parse_manifest(skill_md)
                if manifest is not None:
                    self._manifests.setdefault(manifest.name, manifest)

    def get(self, name: str) -> SkillManifest | None:
        return self._manifests.get(name)

    def list(self) -> list[SkillManifest]:
        return sorted(self._manifests.values(), key=lambda m: m.name)

    def _parse_manifest(self, skill_md: Path) -> SkillManifest | None:
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            return None
        meta, _ = parse_frontmatter(text)
        name = meta.get("name", "").strip()
        if not name:
            return None
        description = meta.get("description", "").strip()
        # frontmatter 中 name/description 之外的键保留为 extra，保持扩展性
        extra = {k: v for k, v in meta.items() if k not in _RESERVED_KEYS}
        skill_dir = skill_md.parent
        # 同级目录下除 SKILL.md 之外的所有文件视为技能资源（模板、示例等）
        resources = [
            path
            for path in sorted(skill_dir.rglob("*"))
            if path.is_file() and path != skill_md
        ]
        return SkillManifest(
            name=name,
            description=description,
            path=skill_dir,
            body_path=skill_md,
            extra=extra,
            resources=resources,
        )
