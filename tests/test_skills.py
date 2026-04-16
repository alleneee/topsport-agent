from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.skills import (
    SkillInjector,
    SkillLoader,
    SkillMatcher,
    SkillRegistry,
    build_skill_tools,
)
from topsport_agent.types.message import Message, ToolCall
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


def _write_skill(
    base: Path,
    name: str,
    description: str,
    body: str,
    **extra: str,
) -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    for key, value in extra.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    (skill_dir / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    return skill_dir


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    base = tmp_path / "skills"
    base.mkdir()
    _write_skill(
        base,
        "echo-helper",
        "Use when the user asks to echo back text.",
        "# Echo Helper\n\nJust echo the input back to the user verbatim.",
    )
    _write_skill(
        base,
        "math-helper",
        "Use when the user needs arithmetic computations.",
        "# Math Helper\n\nCompute the result step by step.",
        version="1.0.0",
    )
    return base


@pytest.fixture
def registry(skill_dir: Path) -> SkillRegistry:
    reg = SkillRegistry([skill_dir])
    reg.load()
    return reg


@pytest.fixture
def cancel_event() -> asyncio.Event:
    return asyncio.Event()


def _ctx(session_id: str, cancel_event: asyncio.Event) -> ToolContext:
    return ToolContext(
        session_id=session_id, call_id="c1", cancel_event=cancel_event
    )


def test_registry_loads_manifests(registry: SkillRegistry) -> None:
    names = [manifest.name for manifest in registry.list()]
    assert names == ["echo-helper", "math-helper"]

    math = registry.get("math-helper")
    assert math is not None
    assert math.description == "Use when the user needs arithmetic computations."
    assert math.extra == {"version": "1.0.0"}
    assert math.body_path.name == "SKILL.md"


def test_registry_get_unknown_returns_none(registry: SkillRegistry) -> None:
    assert registry.get("does-not-exist") is None


def test_registry_skips_missing_name(tmp_path: Path) -> None:
    base = tmp_path / "skills"
    base.mkdir()
    (base / "broken").mkdir()
    (base / "broken" / "SKILL.md").write_text(
        "---\ndescription: no name field\n---\n\nbody",
        encoding="utf-8",
    )
    reg = SkillRegistry([base])
    reg.load()
    assert reg.list() == []


def test_loader_returns_body_without_frontmatter(registry: SkillRegistry) -> None:
    loader = SkillLoader(registry)
    loaded = loader.load("echo-helper")

    assert loaded is not None
    assert loaded.manifest.name == "echo-helper"
    assert loaded.body.startswith("# Echo Helper")
    assert "---" not in loaded.body.split("\n", 1)[0]
    assert "Just echo the input" in loaded.body


def test_loader_unknown_skill_returns_none(registry: SkillRegistry) -> None:
    loader = SkillLoader(registry)
    assert loader.load("does-not-exist") is None


def test_matcher_activate_and_deactivate(registry: SkillRegistry) -> None:
    matcher = SkillMatcher(registry)
    assert matcher.activate("s1", "echo-helper") is True
    assert matcher.active_skills("s1") == ["echo-helper"]

    assert matcher.deactivate("s1", "echo-helper") is True
    assert matcher.active_skills("s1") == []

    assert matcher.deactivate("s1", "echo-helper") is False


def test_matcher_rejects_unknown_skill(registry: SkillRegistry) -> None:
    matcher = SkillMatcher(registry)
    assert matcher.activate("s1", "nope") is False
    assert matcher.active_skills("s1") == []


def test_matcher_session_scoped(registry: SkillRegistry) -> None:
    matcher = SkillMatcher(registry)
    matcher.activate("s1", "echo-helper")
    matcher.activate("s2", "math-helper")

    assert matcher.active_skills("s1") == ["echo-helper"]
    assert matcher.active_skills("s2") == ["math-helper"]


async def test_injector_emits_catalog_only_when_no_skill_active(
    registry: SkillRegistry,
) -> None:
    loader = SkillLoader(registry)
    matcher = SkillMatcher(registry)
    injector = SkillInjector(registry, loader, matcher)

    messages = await injector.provide(Session(id="s1", system_prompt="sys"))

    assert len(messages) == 1
    content = messages[0].content or ""
    assert "Available skills" in content
    assert "- `echo-helper`:" in content
    assert "- `math-helper`:" in content
    assert "load_skill" in content


async def test_injector_adds_active_skill_body(registry: SkillRegistry) -> None:
    loader = SkillLoader(registry)
    matcher = SkillMatcher(registry)
    injector = SkillInjector(registry, loader, matcher)
    matcher.activate("s1", "echo-helper")

    messages = await injector.provide(Session(id="s1", system_prompt="sys"))

    assert len(messages) == 2
    active_msg = messages[1]
    content = active_msg.content or ""
    assert "## Skill: echo-helper" in content
    assert "Just echo the input" in content


async def test_injector_without_catalog(registry: SkillRegistry) -> None:
    loader = SkillLoader(registry)
    matcher = SkillMatcher(registry)
    injector = SkillInjector(registry, loader, matcher, include_catalog=False)

    messages = await injector.provide(Session(id="s1", system_prompt="sys"))
    assert messages == []


async def test_load_skill_tool_activates(
    registry: SkillRegistry, cancel_event: asyncio.Event
) -> None:
    matcher = SkillMatcher(registry)
    tools = {tool.name: tool for tool in build_skill_tools(registry, matcher)}

    result = await tools["load_skill"].handler(
        {"name": "echo-helper"}, _ctx("s1", cancel_event)
    )
    assert result["ok"] is True
    assert result["name"] == "echo-helper"
    assert matcher.active_skills("s1") == ["echo-helper"]


async def test_load_skill_tool_unknown_name(
    registry: SkillRegistry, cancel_event: asyncio.Event
) -> None:
    matcher = SkillMatcher(registry)
    tools = {tool.name: tool for tool in build_skill_tools(registry, matcher)}

    result = await tools["load_skill"].handler(
        {"name": "nope"}, _ctx("s1", cancel_event)
    )
    assert result["ok"] is False
    assert "nope" in result["error"]
    assert "echo-helper" in result["available"]


async def test_unload_skill_tool(
    registry: SkillRegistry, cancel_event: asyncio.Event
) -> None:
    matcher = SkillMatcher(registry)
    matcher.activate("s1", "echo-helper")
    tools = {tool.name: tool for tool in build_skill_tools(registry, matcher)}

    result = await tools["unload_skill"].handler(
        {"name": "echo-helper"}, _ctx("s1", cancel_event)
    )
    assert result == {"ok": True, "name": "echo-helper"}
    assert matcher.active_skills("s1") == []


async def test_list_skills_tool(
    registry: SkillRegistry, cancel_event: asyncio.Event
) -> None:
    matcher = SkillMatcher(registry)
    matcher.activate("s1", "math-helper")
    tools = {tool.name: tool for tool in build_skill_tools(registry, matcher)}

    result = await tools["list_skills"].handler({}, _ctx("s1", cancel_event))
    assert result["count"] == 2
    by_name = {item["name"]: item for item in result["skills"]}
    assert by_name["echo-helper"]["active"] is False
    assert by_name["math-helper"]["active"] is True


class _ScriptedProvider:
    name = "scripted"

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.seen_calls: list[list[Message]] = []
        self.seen_tools: list[list[ToolSpec]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.seen_calls.append(list(request.messages))
        self.seen_tools.append(list(request.tools))
        if self._index >= len(self._turns):
            return LLMResponse(text="fallback", finish_reason="stop")
        turn = self._turns[self._index]
        self._index += 1
        return turn


async def test_skill_activation_visible_on_next_step_through_engine(
    registry: SkillRegistry,
) -> None:
    loader = SkillLoader(registry)
    matcher = SkillMatcher(registry)
    injector = SkillInjector(registry, loader, matcher)
    skill_tools = build_skill_tools(registry, matcher)

    provider = _ScriptedProvider(
        [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="load_skill",
                        arguments={"name": "echo-helper"},
                    )
                ],
                finish_reason="tool_use",
            ),
            LLMResponse(text="acknowledged", finish_reason="stop"),
        ]
    )
    engine = Engine(
        provider,
        tools=skill_tools,
        config=EngineConfig(model="fake"),
        context_providers=[injector],
    )
    session = Session(id="skill-sess", system_prompt="sys")

    async for _ in engine.run(session):
        pass

    assert session.state == RunState.DONE
    assert matcher.active_skills("skill-sess") == ["echo-helper"]

    first_call_system = provider.seen_calls[0][0].content or ""
    assert "Available skills" in first_call_system
    assert "## Skill: echo-helper" not in first_call_system

    second_call_system = provider.seen_calls[1][0].content or ""
    assert "## Skill: echo-helper" in second_call_system
    assert "Just echo the input" in second_call_system


_CLAUDE_SKILLS_DIR = Path.home() / ".claude" / "skills"


@pytest.mark.skipif(
    not _CLAUDE_SKILLS_DIR.exists(),
    reason="Claude skills directory not present on this machine",
)
def test_registry_parses_real_claude_official_skills() -> None:
    reg = SkillRegistry([_CLAUDE_SKILLS_DIR])
    reg.load()

    manifests = reg.list()
    assert len(manifests) >= 3

    for manifest in manifests:
        assert manifest.name, f"skill at {manifest.body_path} missing name"
        assert manifest.description, f"skill {manifest.name} missing description"
        assert manifest.body_path.name == "SKILL.md"
        assert manifest.body_path.exists()
