"""Agent 抽象层测试：AgentConfig + Agent.from_config + default_agent + browser_agent。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.agent import (
    BROWSER_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    Agent,
    AgentConfig,
    BrowserUnavailableError,
    browser_agent,
    default_agent,
    extract_assistant_text,
)
from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.message import Message, Role
from topsport_agent.types.session import Session

# ---------------------------------------------------------------------------
# 测试用 Mock Provider
# ---------------------------------------------------------------------------


@dataclass
class MockProvider:
    """最小 LLMProvider mock：返回一条预设的 assistant 文本。"""

    response_text: str = "hello from mock"
    calls: list[LLMRequest] = field(default_factory=list)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self.response_text,
            tool_calls=[],
            finish_reason="stop",
            usage={"input_tokens": 10, "output_tokens": 5},
            response_metadata=None,
        )


# ---------------------------------------------------------------------------
# AgentConfig 基础
# ---------------------------------------------------------------------------


def test_agent_config_defaults() -> None:
    """默认值：skills/memory/plugins 开启，browser 关闭。"""
    cfg = AgentConfig(
        name="test",
        description="test agent",
        system_prompt="you are X",
        model="any/model",
    )
    assert cfg.enable_skills is True
    assert cfg.enable_memory is True
    assert cfg.enable_plugins is True
    assert cfg.enable_browser is False
    assert cfg.max_steps == 20


def test_agent_config_extras_are_independent_lists() -> None:
    """默认工厂产出独立列表（slots=True + field default_factory）。"""
    cfg_a = AgentConfig(name="a", description="", system_prompt="", model="m")
    cfg_b = AgentConfig(name="b", description="", system_prompt="", model="m")
    cfg_a.extra_tools.append(None)  # type: ignore[arg-type]
    assert cfg_b.extra_tools == []


# ---------------------------------------------------------------------------
# Agent.from_config 组装
# ---------------------------------------------------------------------------


def test_agent_from_config_minimal(tmp_path: Path) -> None:
    """关掉所有能力时，Agent 只含 extra_tools。"""
    provider = MockProvider()
    config = AgentConfig(
        name="bare",
        description="",
        system_prompt="test",
        model="m",
        enable_skills=False,
        enable_memory=False,
        enable_plugins=False,
        enable_browser=False,
    )
    agent = Agent.from_config(provider, config)  # type: ignore[arg-type]
    assert agent.config.name == "bare"
    assert agent.skill_registry is None
    assert agent.plugin_manager is None


def test_agent_from_config_with_skills(tmp_path: Path) -> None:
    """启用 skills 时 skill_registry 被填充。"""
    provider = MockProvider()
    config = AgentConfig(
        name="skills",
        description="",
        system_prompt="",
        model="m",
        enable_skills=True,
        enable_memory=False,
        enable_plugins=False,
        enable_browser=False,
        local_skill_dirs=[tmp_path / "skills"],
    )
    agent = Agent.from_config(provider, config)  # type: ignore[arg-type]
    assert agent.skill_registry is not None


def test_agent_from_config_with_plugins(tmp_path: Path) -> None:
    """启用 plugins 时 plugin_manager 被填充，指向空目录时不炸。"""
    provider = MockProvider()
    config = AgentConfig(
        name="plugins",
        description="",
        system_prompt="",
        model="m",
        enable_skills=False,
        enable_memory=False,
        enable_plugins=True,
        enable_browser=False,
    )
    agent = Agent.from_config(provider, config)  # type: ignore[arg-type]
    assert agent.plugin_manager is not None


# ---------------------------------------------------------------------------
# new_session / run / close
# ---------------------------------------------------------------------------


def test_agent_new_session_binds_system_prompt() -> None:
    provider = MockProvider()
    config = AgentConfig(
        name="t", description="", system_prompt="SP", model="m",
        enable_skills=False, enable_memory=False, enable_plugins=False,
    )
    agent = Agent.from_config(provider, config)  # type: ignore[arg-type]
    session = agent.new_session()
    assert session.system_prompt == "SP"
    assert session.id  # 自动生成


def test_agent_new_session_with_explicit_id() -> None:
    provider = MockProvider()
    config = AgentConfig(
        name="t", description="", system_prompt="SP", model="m",
        enable_skills=False, enable_memory=False, enable_plugins=False,
    )
    agent = Agent.from_config(provider, config)  # type: ignore[arg-type]
    session = agent.new_session("custom-id")
    assert session.id == "custom-id"


async def test_agent_run_appends_user_message_and_runs() -> None:
    """run() 把 user 输入加入 session 并驱动 engine。"""
    provider = MockProvider(response_text="assistant reply")
    config = AgentConfig(
        name="t", description="", system_prompt="SP", model="m",
        enable_skills=False, enable_memory=False, enable_plugins=False,
    )
    agent = Agent.from_config(provider, config)  # type: ignore[arg-type]
    session = agent.new_session()

    events: list[Event] = []
    async for event in agent.run("hi", session):
        events.append(event)

    # user + assistant 消息都应落盘
    assert session.messages[0].role == Role.USER
    assert session.messages[0].content == "hi"
    assert any(m.role == Role.ASSISTANT for m in session.messages)
    # provider 被调用了一次
    assert len(provider.calls) == 1


async def test_agent_close_runs_cleanup_callbacks() -> None:
    """close() 按注册顺序执行清理回调。"""
    provider = MockProvider()
    config = AgentConfig(
        name="t", description="", system_prompt="", model="m",
        enable_skills=False, enable_memory=False, enable_plugins=False,
    )
    agent = Agent.from_config(provider, config)  # type: ignore[arg-type]

    calls: list[str] = []

    async def cb1() -> None:
        calls.append("cb1")

    async def cb2() -> None:
        calls.append("cb2")

    agent._cleanup_callbacks.extend([cb1, cb2])
    await agent.close()
    assert calls == ["cb1", "cb2"]


async def test_agent_close_swallows_exceptions() -> None:
    """单个回调抛异常不中断清理链。"""
    provider = MockProvider()
    config = AgentConfig(
        name="t", description="", system_prompt="", model="m",
        enable_skills=False, enable_memory=False, enable_plugins=False,
    )
    agent = Agent.from_config(provider, config)  # type: ignore[arg-type]

    calls: list[str] = []

    async def cb_bad() -> None:
        raise RuntimeError("boom")

    async def cb_good() -> None:
        calls.append("good")

    agent._cleanup_callbacks.extend([cb_bad, cb_good])
    await agent.close()
    assert calls == ["good"]


# ---------------------------------------------------------------------------
# default_agent 工厂
# ---------------------------------------------------------------------------


def test_default_agent_uses_default_system_prompt(tmp_path: Path) -> None:
    """默认 agent 使用 DEFAULT_SYSTEM_PROMPT。"""
    provider = MockProvider()
    agent = default_agent(
        provider=provider,  # type: ignore[arg-type]
        model="m",
        enable_browser=False,
        memory_base_path=tmp_path / "mem",
        local_skill_dirs=[tmp_path / "skills"],
    )
    assert agent.config.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert "list_skills" in agent.config.system_prompt


def test_default_agent_custom_system_prompt(tmp_path: Path) -> None:
    provider = MockProvider()
    agent = default_agent(
        provider=provider,  # type: ignore[arg-type]
        model="m",
        system_prompt="custom!",
        enable_browser=False,
        memory_base_path=tmp_path / "mem",
        local_skill_dirs=[tmp_path / "skills"],
    )
    assert agent.config.system_prompt == "custom!"


def test_default_agent_extras_merged(tmp_path: Path) -> None:
    """extra_tools 被并入最终 tool 池。"""
    from topsport_agent.types.tool import ToolSpec

    async def handler(args: dict, ctx: Any) -> dict:
        return {"ok": True}

    my_tool = ToolSpec(
        name="my_custom_tool",
        description="",
        parameters={"type": "object", "properties": {}},
        handler=handler,  # type: ignore[arg-type]
    )

    provider = MockProvider()
    agent = default_agent(
        provider=provider,  # type: ignore[arg-type]
        model="m",
        enable_browser=False,
        memory_base_path=tmp_path / "mem",
        local_skill_dirs=[tmp_path / "skills"],
        extra_tools=[my_tool],
    )
    tool_names = [t.name for t in agent.engine._tools]
    assert "my_custom_tool" in tool_names


# ---------------------------------------------------------------------------
# browser_agent 工厂
# ---------------------------------------------------------------------------


def test_browser_agent_raises_when_playwright_missing(tmp_path: Path) -> None:
    """Playwright 未装时抛 BrowserUnavailableError。"""
    import importlib.util

    if importlib.util.find_spec("playwright") is not None:
        pytest.skip("playwright installed — skip missing-playwright test")

    provider = MockProvider()
    with pytest.raises(BrowserUnavailableError):
        browser_agent(
            provider=provider,  # type: ignore[arg-type]
            model="m",
            memory_base_path=tmp_path / "mem",
            local_skill_dirs=[tmp_path / "skills"],
        )


def test_browser_agent_has_specialized_prompt() -> None:
    """browser_agent 的默认 prompt 含 browser_navigate 等工具说明。"""
    # 直接验证 BROWSER_SYSTEM_PROMPT 内容即可，不必实例化（避免 playwright 依赖）
    assert "browser_navigate" in BROWSER_SYSTEM_PROMPT
    assert "browser_snapshot" in BROWSER_SYSTEM_PROMPT
    assert "@ref" in BROWSER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# extract_assistant_text
# ---------------------------------------------------------------------------


def test_extract_assistant_text_returns_last_reply() -> None:
    session = Session(id="s", system_prompt="")
    session.messages.append(Message(role=Role.USER, content="hi"))
    session.messages.append(Message(role=Role.ASSISTANT, content="hello"))
    events = [
        Event(type=EventType.MESSAGE_APPENDED, session_id="s", payload={"role": "assistant"}),
    ]
    result = extract_assistant_text(events, session)
    assert result == "hello"


def test_extract_assistant_text_none_when_no_reply() -> None:
    session = Session(id="s", system_prompt="")
    session.messages.append(Message(role=Role.USER, content="hi"))
    events = [
        Event(type=EventType.MESSAGE_APPENDED, session_id="s", payload={"role": "user"}),
    ]
    result = extract_assistant_text(events, session)
    assert result is None
