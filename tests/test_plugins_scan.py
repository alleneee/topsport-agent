"""插件扫描测试：单元测试 + 真实插件集成测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from topsport_agent.plugins.discovery import InstalledPlugin, discover_plugins
from topsport_agent.plugins.plugin import PluginDescriptor, scan_plugin


def _make_installed(tmp_path: Path, name: str = "test-plugin") -> InstalledPlugin:
    return InstalledPlugin(
        name=name,
        marketplace="test-mp",
        install_path=tmp_path,
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------


def test_scan_empty_plugin(tmp_path: Path) -> None:
    """空插件目录：四种扩展都为空。"""
    result = scan_plugin(_make_installed(tmp_path))
    assert result.skill_dirs == []
    assert result.command_paths == []
    assert result.agent_paths == []
    assert result.hooks_config is None


def test_scan_skills(tmp_path: Path) -> None:
    """扫描 skills/*/SKILL.md 目录。"""
    skill_a = tmp_path / "skills" / "alpha"
    skill_b = tmp_path / "skills" / "beta"
    skill_a.mkdir(parents=True)
    skill_b.mkdir(parents=True)
    (skill_a / "SKILL.md").write_text("---\nname: alpha\n---\nbody", encoding="utf-8")
    (skill_b / "SKILL.md").write_text("---\nname: beta\n---\nbody", encoding="utf-8")
    # 没有 SKILL.md 的目录应被跳过
    (tmp_path / "skills" / "no-skill").mkdir()

    result = scan_plugin(_make_installed(tmp_path))
    assert len(result.skill_dirs) == 2
    assert result.skill_dirs[0].name == "alpha"
    assert result.skill_dirs[1].name == "beta"


def test_scan_commands(tmp_path: Path) -> None:
    """扫描 commands/*.md 文件。"""
    cmd_dir = tmp_path / "commands"
    cmd_dir.mkdir()
    (cmd_dir / "build.md").write_text("---\ndescription: build\n---\nbody")
    (cmd_dir / "deploy.md").write_text("---\ndescription: deploy\n---\nbody")
    (cmd_dir / "README.txt").write_text("not a command")  # 非 .md 跳过

    result = scan_plugin(_make_installed(tmp_path))
    assert len(result.command_paths) == 2
    assert result.command_paths[0].name == "build.md"
    assert result.command_paths[1].name == "deploy.md"


def test_scan_agents(tmp_path: Path) -> None:
    """扫描 agents/*.md 文件。"""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "reviewer.md").write_text("---\nname: reviewer\n---\nbody")

    result = scan_plugin(_make_installed(tmp_path))
    assert len(result.agent_paths) == 1
    assert result.agent_paths[0].name == "reviewer.md"


def test_scan_hooks(tmp_path: Path) -> None:
    """发现 hooks/hooks.json。"""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.json").write_text('{"hooks": {}}')

    result = scan_plugin(_make_installed(tmp_path))
    assert result.hooks_config is not None
    assert result.hooks_config.name == "hooks.json"


def test_scan_no_hooks(tmp_path: Path) -> None:
    """hooks 目录存在但没有 hooks.json。"""
    (tmp_path / "hooks").mkdir()

    result = scan_plugin(_make_installed(tmp_path))
    assert result.hooks_config is None


def test_scan_all_types(tmp_path: Path) -> None:
    """一个插件同时有四种扩展。"""
    # skill
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n")
    # command
    cmd_dir = tmp_path / "commands"
    cmd_dir.mkdir()
    (cmd_dir / "run.md").write_text("---\ndescription: run\n---\n")
    # agent
    agent_dir = tmp_path / "agents"
    agent_dir.mkdir()
    (agent_dir / "helper.md").write_text("---\nname: helper\n---\n")
    # hooks
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "hooks.json").write_text('{"hooks": {}}')

    result = scan_plugin(_make_installed(tmp_path))
    assert len(result.skill_dirs) == 1
    assert len(result.command_paths) == 1
    assert len(result.agent_paths) == 1
    assert result.hooks_config is not None


# ---------------------------------------------------------------------------
# 集成测试：真实插件
# ---------------------------------------------------------------------------

_REAL_PLUGINS_DIR = Path.home() / ".claude" / "plugins"


@pytest.mark.skipif(
    not (_REAL_PLUGINS_DIR / "installed_plugins.json").is_file(),
    reason="~/.claude/plugins/installed_plugins.json not found",
)
class TestRealPluginScan:
    def test_scan_superpowers_has_skills(self) -> None:
        """superpowers 插件应有 skills 目录。"""
        plugins = discover_plugins(_REAL_PLUGINS_DIR)
        sp = next((p for p in plugins if p.name == "superpowers"), None)
        assert sp is not None
        desc = scan_plugin(sp)
        assert len(desc.skill_dirs) > 0

    def test_scan_superpowers_has_agents(self) -> None:
        """superpowers 插件应有 agents 目录。"""
        plugins = discover_plugins(_REAL_PLUGINS_DIR)
        sp = next((p for p in plugins if p.name == "superpowers"), None)
        assert sp is not None
        desc = scan_plugin(sp)
        assert len(desc.agent_paths) > 0

    def test_scan_superpowers_has_hooks(self) -> None:
        """superpowers 插件应有 hooks.json。"""
        plugins = discover_plugins(_REAL_PLUGINS_DIR)
        sp = next((p for p in plugins if p.name == "superpowers"), None)
        assert sp is not None
        desc = scan_plugin(sp)
        assert desc.hooks_config is not None

    def test_scan_all_plugins_no_crash(self) -> None:
        """扫描所有真实插件不应抛异常。"""
        plugins = discover_plugins(_REAL_PLUGINS_DIR)
        for plugin in plugins:
            desc = scan_plugin(plugin)
            assert isinstance(desc, PluginDescriptor)
