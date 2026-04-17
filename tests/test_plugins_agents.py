"""Agent 注册表测试：解析、注册、工具调用。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from topsport_agent.plugins.agent_registry import (
    AgentDefinition,
    AgentRegistry,
    _parse_agent_md,
    build_agent_tools,
)
from topsport_agent.plugins.discovery import InstalledPlugin, discover_plugins
from topsport_agent.plugins.plugin import PluginDescriptor, scan_plugin
from topsport_agent.types.tool import ToolContext

# ---------------------------------------------------------------------------
# _parse_agent_md 单元测试
# ---------------------------------------------------------------------------




def test_parse_agent_basic(tmp_path: Path) -> None:
    """解析基本的 agent md 文件。"""
    md = tmp_path / "reviewer.md"
    md.write_text(
        "---\n"
        "name: reviewer\n"
        "description: Reviews code quality\n"
        "model: sonnet\n"
        "---\n"
        "\n"
        "You are a code reviewer.\n",
        encoding="utf-8",
    )
    result = _parse_agent_md(md, "test-plugin")
    assert result is not None
    assert result.name == "reviewer"
    assert result.qualified_name == "test-plugin:reviewer"
    assert result.description == "Reviews code quality"
    assert result.model == "sonnet"
    assert result.body == "You are a code reviewer."


def test_parse_agent_with_tools(tmp_path: Path) -> None:
    """解析带 tools 字段的 agent。"""
    md = tmp_path / "helper.md"
    md.write_text(
        "---\n"
        "name: helper\n"
        "description: Helps with tasks\n"
        "tools: Read, Glob, Grep, Bash\n"
        "---\n"
        "\n"
        "body\n",
        encoding="utf-8",
    )
    result = _parse_agent_md(md, "p")
    assert result is not None
    assert result.allowed_tools == ["Read", "Glob", "Grep", "Bash"]


def test_parse_agent_with_skills(tmp_path: Path) -> None:
    """解析带 skills 字段的 agent（多行 YAML list 格式）。"""
    md = tmp_path / "agent.md"
    md.write_text(
        "---\n"
        "name: agent\n"
        "description: desc\n"
        "skills: |\n"
        "  skill-a\n"
        "  skill-b\n"
        "---\n"
        "\n"
        "body\n",
        encoding="utf-8",
    )
    result = _parse_agent_md(md, "p")
    assert result is not None
    assert "skill-a" in result.auto_skills
    assert "skill-b" in result.auto_skills


def test_parse_agent_name_from_filename(tmp_path: Path) -> None:
    """无 name frontmatter 时从文件名推导。"""
    md = tmp_path / "my-agent.md"
    md.write_text(
        "---\n"
        "description: no name field\n"
        "---\n"
        "\n"
        "body\n",
        encoding="utf-8",
    )
    result = _parse_agent_md(md, "p")
    assert result is not None
    assert result.name == "my-agent"
    assert result.qualified_name == "p:my-agent"


def test_parse_agent_default_model(tmp_path: Path) -> None:
    """无 model 字段时默认 inherit。"""
    md = tmp_path / "a.md"
    md.write_text("---\nname: a\ndescription: d\n---\nbody\n")
    result = _parse_agent_md(md, "p")
    assert result is not None
    assert result.model == "inherit"


def test_parse_agent_missing_file(tmp_path: Path) -> None:
    """文件不存在时返回 None。"""
    result = _parse_agent_md(tmp_path / "nonexistent.md", "p")
    assert result is None


# ---------------------------------------------------------------------------
# AgentRegistry 单元测试
# ---------------------------------------------------------------------------


def test_registry_register_and_get() -> None:
    reg = AgentRegistry()
    agent = AgentDefinition(
        name="a", qualified_name="p:a", description="d", body="b"
    )
    reg.register(agent)
    assert reg.get("p:a") is agent
    assert reg.get("nonexistent") is None


def test_registry_dedup_first_wins() -> None:
    """同名 agent 先注册优先。"""
    reg = AgentRegistry()
    first = AgentDefinition(name="a", qualified_name="p:a", description="first", body="1")
    second = AgentDefinition(name="a", qualified_name="p:a", description="second", body="2")
    reg.register(first)
    reg.register(second)
    assert reg.get("p:a").description == "first"  # type: ignore[union-attr]


def test_registry_list_sorted() -> None:
    reg = AgentRegistry()
    reg.register(AgentDefinition(name="z", qualified_name="p:z", description="", body=""))
    reg.register(AgentDefinition(name="a", qualified_name="p:a", description="", body=""))
    result = reg.list()
    assert result[0].qualified_name == "p:a"
    assert result[1].qualified_name == "p:z"


def test_registry_load_from_plugins(tmp_path: Path) -> None:
    """从 PluginDescriptor 加载 agents。"""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "bot.md").write_text(
        "---\nname: bot\ndescription: a bot\n---\nbot body\n"
    )

    desc = PluginDescriptor(
        info=InstalledPlugin(name="myplugin", marketplace="mp", install_path=tmp_path, version="1.0"),
        agent_paths=[agents_dir / "bot.md"],
    )
    reg = AgentRegistry()
    reg.load_from_plugins([desc])
    assert reg.get("myplugin:bot") is not None
    assert reg.get("myplugin:bot").body == "bot body"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# build_agent_tools 测试
# ---------------------------------------------------------------------------


def _make_ctx() -> ToolContext:
    return ToolContext(session_id="test-session", call_id="call-1", cancel_event=asyncio.Event())


async def test_list_agents_tool() -> None:
    reg = AgentRegistry()
    reg.register(AgentDefinition(
        name="r", qualified_name="p:r", description="reviewer", body="b", model="opus"
    ))
    tools = build_agent_tools(reg)
    list_tool = next(t for t in tools if t.name == "list_agents")
    result = await list_tool.handler({}, _make_ctx())
    assert result["count"] == 1
    assert result["agents"][0]["name"] == "p:r"
    assert result["agents"][0]["model"] == "opus"


async def test_spawn_agent_found() -> None:
    reg = AgentRegistry()
    reg.register(AgentDefinition(
        name="helper", qualified_name="p:helper", description="helps", body="system prompt"
    ))
    tools = build_agent_tools(reg)
    spawn_tool = next(t for t in tools if t.name == "spawn_agent")
    result = await spawn_tool.handler({"name": "p:helper", "task": "do stuff"}, _make_ctx())
    assert result["ok"] is True
    assert result["system_prompt"] == "system prompt"
    assert result["task"] == "do stuff"


async def test_spawn_agent_not_found() -> None:
    reg = AgentRegistry()
    tools = build_agent_tools(reg)
    spawn_tool = next(t for t in tools if t.name == "spawn_agent")
    result = await spawn_tool.handler({"name": "ghost", "task": "t"}, _make_ctx())
    assert result["ok"] is False
    assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# 集成测试：真实插件
# ---------------------------------------------------------------------------

_REAL_PLUGINS_DIR = Path.home() / ".claude" / "plugins"


@pytest.mark.skipif(
    not (_REAL_PLUGINS_DIR / "installed_plugins.json").is_file(),
    reason="~/.claude/plugins/installed_plugins.json not found",
)
class TestRealAgents:
    def test_superpowers_agents_parsed(self) -> None:
        """superpowers 插件的 agents 能被正确解析。"""
        plugins = discover_plugins(_REAL_PLUGINS_DIR)
        sp = next((p for p in plugins if p.name == "superpowers"), None)
        assert sp is not None
        desc = scan_plugin(sp)
        reg = AgentRegistry()
        reg.load_from_plugins([desc])
        agents = reg.list()
        assert len(agents) > 0
        # superpowers 有 code-reviewer agent
        names = {a.qualified_name for a in agents}
        assert "superpowers:code-reviewer" in names

    def test_all_plugin_agents_parseable(self) -> None:
        """所有真实插件的 agents 都能被解析，不抛异常。"""
        plugins = discover_plugins(_REAL_PLUGINS_DIR)
        descs = [scan_plugin(p) for p in plugins]
        reg = AgentRegistry()
        reg.load_from_plugins(descs)
        # 只要不抛异常就算通过
        assert isinstance(reg.list(), list)
