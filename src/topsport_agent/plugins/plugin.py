"""插件扫描：遍历插件根目录，分类四种扩展类型的路径。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .discovery import InstalledPlugin


@dataclass(slots=True)
class PluginDescriptor:
    """一个已安装插件的完整描述。"""

    info: InstalledPlugin
    skill_dirs: list[Path] = field(default_factory=list)
    command_paths: list[Path] = field(default_factory=list)
    agent_paths: list[Path] = field(default_factory=list)
    hooks_config: Path | None = None


def scan_plugin(installed: InstalledPlugin) -> PluginDescriptor:
    """扫描插件根目录，按约定路径收集四种扩展。"""
    root = installed.install_path
    descriptor = PluginDescriptor(info=installed)

    # skills/*/SKILL.md
    skills_dir = root / "skills"
    if skills_dir.is_dir():
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                descriptor.skill_dirs.append(child)

    # commands/*.md
    commands_dir = root / "commands"
    if commands_dir.is_dir():
        for child in sorted(commands_dir.iterdir()):
            if child.is_file() and child.suffix == ".md":
                descriptor.command_paths.append(child)

    # agents/*.md
    agents_dir = root / "agents"
    if agents_dir.is_dir():
        for child in sorted(agents_dir.iterdir()):
            if child.is_file() and child.suffix == ".md":
                descriptor.agent_paths.append(child)

    # hooks/hooks.json
    hooks_json = root / "hooks" / "hooks.json"
    if hooks_json.is_file():
        descriptor.hooks_config = hooks_json

    return descriptor
