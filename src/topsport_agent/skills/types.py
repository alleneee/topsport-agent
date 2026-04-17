from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SkillManifest:
    """SkillManifest 是磁盘扫描的产物，只含元数据和路径，不持有文件内容。

    body_path 指向 SKILL.md，resources 收集同级目录下的附属文件。
    """
    name: str
    description: str
    path: Path
    body_path: Path
    extra: dict[str, str] = field(default_factory=dict)
    resources: list[Path] = field(default_factory=list)


@dataclass(slots=True)
class LoadedSkill:
    """LoadedSkill 是 SkillLoader 的输出：manifest + 实际 body 文本 + 资源相对路径索引。"""
    manifest: SkillManifest
    body: str
    resources_index: list[str] = field(default_factory=list)
