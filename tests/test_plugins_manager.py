"""PluginManager 集成测试：端到端 + 真实插件。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from topsport_agent.plugins.manager import PluginManager
from topsport_agent.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# 单元测试：临时目录构造完整假插件
# ---------------------------------------------------------------------------


def _setup_fake_plugin(tmp_path: Path) -> Path:
    """构造一个包含四种扩展的假插件。"""
    plugin_dir = tmp_path / "cache" / "test-mp" / "fake-plugin" / "1.0"
    plugin_dir.mkdir(parents=True)

    # skill
    skill_dir = plugin_dir / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A test skill\n---\n\nSkill body here.\n"
    )
    (skill_dir / "helper.md").write_text("Resource file")

    # command
    cmd_dir = plugin_dir / "commands"
    cmd_dir.mkdir()
    (cmd_dir / "do-thing.md").write_text(
        "---\ndescription: Does a thing\n---\n\nCommand instructions.\n"
    )

    # agent
    agent_dir = plugin_dir / "agents"
    agent_dir.mkdir()
    (agent_dir / "bot.md").write_text(
        "---\nname: bot\ndescription: A test bot\nmodel: inherit\n---\n\nYou are a bot.\n"
    )

    # hooks
    hooks_dir = plugin_dir / "hooks"
    hooks_dir.mkdir()
    hooks_data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo loaded", "async": True}]}
            ]
        }
    }
    (hooks_dir / "hooks.json").write_text(json.dumps(hooks_data))

    # installed_plugins.json
    installed = {
        "version": 2,
        "plugins": {
            "fake-plugin@test-mp": [
                {"installPath": str(plugin_dir), "version": "1.0"}
            ]
        },
    }
    (tmp_path / "installed_plugins.json").write_text(json.dumps(installed))

    return tmp_path


def test_manager_load_fake_plugin(tmp_path: Path) -> None:
    """PluginManager 端到端：加载假插件，验证四种扩展都被收集。"""
    plugins_dir = _setup_fake_plugin(tmp_path)
    mgr = PluginManager(plugins_dir)
    try:
        mgr.load()

        # Skills: 应有 fake-plugin:my-skill
        skill_dirs = mgr.skill_dirs()
        assert len(skill_dirs) >= 1  # skill + command

        # 喂给 SkillRegistry 验证可解析
        local_skills = []  # 无本地 skills
        registry = SkillRegistry(local_skills + skill_dirs)
        registry.load()
        manifests = registry.list()
        names = {m.name for m in manifests}
        assert "fake-plugin:my-skill" in names
        assert "fake-plugin:do-thing" in names

        # Agents
        agent_reg = mgr.agent_registry()
        agents = agent_reg.list()
        assert len(agents) == 1
        assert agents[0].qualified_name == "fake-plugin:bot"

        # Hooks
        hook_runner = mgr.hook_runner()
        assert len(hook_runner._hooks) == 1
    finally:
        mgr.cleanup()


def test_manager_skill_resources_copied(tmp_path: Path) -> None:
    """skill 资源文件被复制到临时目录。"""
    plugins_dir = _setup_fake_plugin(tmp_path)
    mgr = PluginManager(plugins_dir)
    try:
        mgr.load()
        skill_dirs = mgr.skill_dirs()
        # 找到 my-skill 的目录
        found = False
        for sd in skill_dirs:
            for child in sd.iterdir():
                if child.name == "fake-plugin:my-skill" and child.is_dir():
                    assert (child / "SKILL.md").is_file()
                    assert (child / "helper.md").is_file()
                    found = True
        assert found, "fake-plugin:my-skill directory not found in skill_dirs"
    finally:
        mgr.cleanup()


def test_manager_local_skills_priority(tmp_path: Path) -> None:
    """本地 skills 排在 plugin skills 前面时，同名本地优先。"""
    # 设置插件
    plugins_dir = _setup_fake_plugin(tmp_path)
    mgr = PluginManager(plugins_dir)
    mgr.load()

    # 创建同名本地 skill
    local_dir = tmp_path / "local_skills"
    local_skill = local_dir / "fake-plugin:my-skill"
    local_skill.mkdir(parents=True)
    (local_skill / "SKILL.md").write_text(
        "---\nname: fake-plugin:my-skill\ndescription: LOCAL version\n---\n\nLocal body.\n"
    )

    # 本地在前，插件在后
    registry = SkillRegistry([local_dir] + mgr.skill_dirs())
    registry.load()

    manifest = registry.get("fake-plugin:my-skill")
    assert manifest is not None
    assert manifest.description == "LOCAL version"
    mgr.cleanup()


def test_manager_empty_plugins_dir(tmp_path: Path) -> None:
    """空目录：load 不抛异常，产出为空。"""
    mgr = PluginManager(tmp_path)
    mgr.load()
    assert mgr.skill_dirs() == []
    assert mgr.agent_registry().list() == []
    assert mgr.hook_runner()._hooks == []


def test_manager_cleanup(tmp_path: Path) -> None:
    """cleanup 清理临时目录。"""
    plugins_dir = _setup_fake_plugin(tmp_path)
    mgr = PluginManager(plugins_dir)
    mgr.load()
    tmp_dirs = list(mgr._tmp_dirs)
    assert len(tmp_dirs) > 0
    for d in tmp_dirs:
        assert d.exists()
    mgr.cleanup()
    for d in tmp_dirs:
        assert not d.exists()


# ---------------------------------------------------------------------------
# 集成测试：真实 ~/.claude/plugins
# ---------------------------------------------------------------------------

_REAL_PLUGINS_DIR = Path.home() / ".claude" / "plugins"


@pytest.mark.skipif(
    not (_REAL_PLUGINS_DIR / "installed_plugins.json").is_file(),
    reason="~/.claude/plugins/installed_plugins.json not found",
)
class TestRealPluginManager:
    def test_load_real_plugins(self) -> None:
        """加载真实插件生态不抛异常。"""
        mgr = PluginManager()
        mgr.load()
        try:
            assert len(mgr.skill_dirs()) > 0
            assert len(mgr.agent_registry().list()) > 0
        finally:
            mgr.cleanup()

    def test_real_skills_parseable_by_registry(self) -> None:
        """真实插件的 skills 能被 SkillRegistry 正确解析。"""
        mgr = PluginManager()
        mgr.load()
        try:
            local_dir = Path.home() / ".claude" / "skills"
            dirs = []
            if local_dir.is_dir():
                dirs.append(local_dir)
            dirs.extend(mgr.skill_dirs())

            registry = SkillRegistry(dirs)
            registry.load()
            manifests = registry.list()
            assert len(manifests) > 0

            # superpowers 的 skills 应以 superpowers: 为前缀
            sp_skills = [m for m in manifests if m.name.startswith("superpowers:")]
            assert len(sp_skills) > 0
        finally:
            mgr.cleanup()

    def test_real_plugin_count(self) -> None:
        """至少加载到已安装插件数量的 skills。"""
        mgr = PluginManager()
        mgr.load()
        try:
            skill_count = len(mgr.skill_dirs())
            # 真实环境下应该有不少 skills 和 agents
            assert skill_count >= 5, f"only {skill_count} skill dirs found"
        finally:
            mgr.cleanup()
