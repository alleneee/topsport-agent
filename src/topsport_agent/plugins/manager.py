"""PluginManager：统一入口，加载所有已安装 Claude Code 插件。

扫描 ~/.claude/plugins/ 下的插件，收集 skills/commands/agents/hooks 四种扩展，
产出可直接注入 Engine 的对象：skill 目录列表、AgentRegistry、PluginHookRunner。
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from ..skills._frontmatter import parse as parse_frontmatter
from .agent_registry import AgentRegistry
from .discovery import discover_plugins
from .hook_runner import PluginHookRunner
from .plugin import PluginDescriptor, scan_plugin

_logger = logging.getLogger(__name__)


class PluginManager:
    """插件生态统一管理器。

    用法:
        mgr = PluginManager()
        mgr.load()
        skill_dirs = mgr.skill_dirs()       # 喂给 SkillRegistry
        agent_reg = mgr.agent_registry()     # 构建 agent 工具
        hook_runner = mgr.hook_runner()      # 注入 Engine event_subscribers
    """

    def __init__(self, plugins_dir: Path | None = None) -> None:
        self._plugins_dir = plugins_dir
        self._plugins: list[PluginDescriptor] = []
        self._skill_dirs: list[Path] = []
        self._agent_registry = AgentRegistry()
        self._hook_runner = PluginHookRunner()
        self._tmp_dirs: list[Path] = []

    def load(self) -> None:
        """完整加载流程：发现 -> 扫描 -> 构建四种扩展。"""
        installed = discover_plugins(self._plugins_dir)
        _logger.info("discovered %d installed plugins", len(installed))

        self._plugins = [scan_plugin(p) for p in installed]

        self._skill_dirs = self._collect_skill_dirs()
        self._agent_registry = AgentRegistry()
        self._agent_registry.load_from_plugins(self._plugins)
        self._hook_runner = PluginHookRunner.from_plugins(self._plugins)

        skill_count = len(self._skill_dirs)
        agent_count = len(self._agent_registry.list())
        hook_count = len(self._hook_runner._hooks)
        _logger.info(
            "loaded: %d skills/commands, %d agents, %d hooks",
            skill_count, agent_count, hook_count,
        )

    def _collect_skill_dirs(self) -> list[Path]:
        """收集所有插件的 skill 目录 + command 转化目录。

        Skills: 插件的 skills/{name}/ 目录中已有 SKILL.md，但需要在 frontmatter 中
        注入 plugin: 前缀的名称。为避免修改原文件，创建临时目录放置带前缀的 SKILL.md。

        Commands: commands/*.md 没有标准 SKILL.md 结构，转化为临时 skill 目录。
        """
        result: list[Path] = []

        for plugin in self._plugins:
            plugin_name = plugin.info.name

            # Skills: 为每个 skill 创建带前缀名称的临时目录
            for skill_dir in plugin.skill_dirs:
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.is_file():
                    continue
                try:
                    text = skill_md.read_text(encoding="utf-8")
                except OSError:
                    continue
                meta, body = parse_frontmatter(text)
                original_name = meta.get("name", "").strip()
                if not original_name:
                    continue

                qualified_name = f"{plugin_name}:{original_name}"
                # 创建临时目录，写入带前缀名称的 SKILL.md
                tmp_dir = Path(tempfile.mkdtemp(prefix=f"topsport_plugin_{plugin_name}_"))
                self._tmp_dirs.append(tmp_dir)
                skill_out_dir = tmp_dir / qualified_name
                skill_out_dir.mkdir(parents=True)
                # 重写 frontmatter 中的 name 为带前缀版本
                new_text = self._rewrite_skill_name(text, qualified_name)
                (skill_out_dir / "SKILL.md").write_text(new_text, encoding="utf-8")
                # 复制资源文件（除 SKILL.md）
                for res in skill_dir.iterdir():
                    if res.is_file() and res.name != "SKILL.md":
                        shutil.copy2(res, skill_out_dir / res.name)
                    elif res.is_dir():
                        shutil.copytree(res, skill_out_dir / res.name, dirs_exist_ok=True)
                result.append(tmp_dir)

            # Commands: 转化为临时 skill 目录
            for cmd_path in plugin.command_paths:
                try:
                    text = cmd_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                meta, body = parse_frontmatter(text)
                cmd_name = cmd_path.stem  # brainstorm.md -> brainstorm
                qualified_name = f"{plugin_name}:{cmd_name}"
                description = meta.get("description", "").strip()

                tmp_dir = Path(tempfile.mkdtemp(prefix=f"topsport_cmd_{plugin_name}_"))
                self._tmp_dirs.append(tmp_dir)
                cmd_skill_dir = tmp_dir / qualified_name
                cmd_skill_dir.mkdir(parents=True)
                # 构造完整 SKILL.md
                skill_text = (
                    f"---\n"
                    f"name: {qualified_name}\n"
                    f"description: {description}\n"
                    f"---\n"
                    f"\n"
                    f"{body}"
                )
                (cmd_skill_dir / "SKILL.md").write_text(skill_text, encoding="utf-8")
                result.append(tmp_dir)

        return result

    @staticmethod
    def _rewrite_skill_name(text: str, new_name: str) -> str:
        """替换 frontmatter 中的 name 字段值。"""
        if not text.startswith("---\n"):
            return text
        end = text.find("\n---\n", 4)
        if end == -1:
            return text
        header = text[4:end]
        body_part = text[end:]
        new_lines: list[str] = []
        for line in header.splitlines():
            if line.startswith("name:"):
                new_lines.append(f"name: {new_name}")
            else:
                new_lines.append(line)
        return "---\n" + "\n".join(new_lines) + body_part

    def skill_dirs(self) -> list[Path]:
        return list(self._skill_dirs)

    def agent_registry(self) -> AgentRegistry:
        return self._agent_registry

    def hook_runner(self) -> PluginHookRunner:
        return self._hook_runner

    def cleanup(self) -> None:
        """清理临时目录。"""
        for tmp_dir in self._tmp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        self._tmp_dirs.clear()
