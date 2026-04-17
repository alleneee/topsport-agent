from __future__ import annotations

from ._frontmatter import parse as parse_frontmatter
from .registry import SkillRegistry
from .types import LoadedSkill


class SkillLoader:
    """Loader 职责：按名称从注册表取 manifest，读取 SKILL.md 正文，剥离 frontmatter 后返回。

    读取是同步的——技能文件通常很小，不值得 asyncio.to_thread 的开销。
    """
    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def load(self, name: str) -> LoadedSkill | None:
        manifest = self._registry.get(name)
        if manifest is None:
            return None
        try:
            text = manifest.body_path.read_text(encoding="utf-8")
        except OSError:
            return None
        _, body = parse_frontmatter(text)
        # 资源索引只暴露技能目录内的相对路径，避免把绝对路径泄漏给运行时
        resources_index = sorted(
            str(path.relative_to(manifest.path)) for path in manifest.resources
        )
        return LoadedSkill(
            manifest=manifest,
            body=body.rstrip("\n"),
            resources_index=resources_index,
        )
