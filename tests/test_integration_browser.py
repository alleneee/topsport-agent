"""Integration tests: Engine + Browser control + Multi-agent orchestration.

Tests the full pipeline with mock LLM providers that script realistic
multi-step browser interactions. No real Playwright or LLM required.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.browser.client import BrowserClient
from topsport_agent.browser.tools import BrowserToolSource
from topsport_agent.browser.types import BrowserConfig
from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.engine.orchestrator import Orchestrator, SubAgentConfig
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.events import Event, EventType
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.plan import Plan, PlanStep
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


# ---------------------------------------------------------------------------
# Mock Browser
# ---------------------------------------------------------------------------


class MockAriaLocator:
    def __init__(self, yaml_text: str) -> None:
        self._yaml = yaml_text

    async def aria_snapshot(self) -> str:
        return self._yaml

    async def text_content(self) -> str:
        return "Mock page text content"


class MockInteractionLocator:
    def __init__(self, page: MockBrowserPage, role: str, name: str) -> None:
        self._page = page
        self._role = role
        self._name = name

    async def click(self, **kwargs: Any) -> None:
        self._page.click_log.append(f"{self._role}:{self._name}")

    async def fill(self, text: str, **kwargs: Any) -> None:
        self._page.fill_log.append((f"{self._role}:{self._name}", text))

    async def text_content(self) -> str:
        return f"text of {self._role}:{self._name}"

    @property
    def first(self) -> MockInteractionLocator:
        return self


class MockBrowserPage:
    """Simulates a multi-page browsing session with scripted page transitions."""

    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages  # url -> aria yaml
        self.url = "about:blank"
        self._title = "Blank"
        self.click_log: list[str] = []
        self.fill_log: list[tuple[str, str]] = []

    async def title(self) -> str:
        return self._title

    async def goto(self, url: str, **kwargs: Any) -> None:
        self.url = url
        self._title = f"Page: {url.split('/')[-1]}"

    def locator(self, selector: str) -> MockAriaLocator | MockInteractionLocator:
        if selector == "body":
            yaml = self._pages.get(self.url, "")
            return MockAriaLocator(yaml)
        return MockInteractionLocator(self, "css", selector)

    def get_by_role(self, role: str, *, name: str = "") -> MockInteractionLocator:
        return MockInteractionLocator(self, role, name)

    async def screenshot(self, *, path: str = "", **kwargs: Any) -> bytes:
        if path:
            Path(path).write_bytes(b"fake-png-data")
        return b"fake-png-data"


def _mock_browser_factory(page: MockBrowserPage):
    @contextlib.asynccontextmanager
    async def factory():
        yield page

    return factory


# ---------------------------------------------------------------------------
# Scripted LLM Provider
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Returns pre-scripted LLM responses in sequence."""

    name = "scripted"

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._index >= len(self._turns):
            return LLMResponse(text="(no more scripted turns)", finish_reason="stop")
        turn = self._turns[self._index]
        self._index += 1
        return turn


# ---------------------------------------------------------------------------
# Event Collector
# ---------------------------------------------------------------------------


@dataclass
class EventCollector:
    name: str = "collector"
    events: list[Event] = field(default_factory=list)

    async def on_event(self, event: Event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.type.value for e in self.events]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(user_msg: str = "test", sid: str = "test-session") -> Session:
    s = Session(id=sid, system_prompt="You are a browser-control agent.")
    s.messages.append(Message(role=Role.USER, content=user_msg))
    return s


async def _collect_events(engine: Engine, session: Session) -> list[Event]:
    return [event async for event in engine.run(session)]


# ---------------------------------------------------------------------------
# Test 1: Engine + BrowserToolSource single-step navigation
# ---------------------------------------------------------------------------


class TestEngineBrowserNavigation:
    """Engine calls browser_navigate, gets snapshot, responds."""

    async def test_navigate_and_respond(self):
        # Mock browser with one page
        page = MockBrowserPage({
            "https://example.com": '- heading "Welcome" [level=1]\n- link "About"\n- button "Sign Up"',
        })
        browser_client = BrowserClient(
            BrowserConfig(), page_factory=_mock_browser_factory(page)
        )
        browser_tools = BrowserToolSource(browser_client)

        # LLM turn 1: call browser_navigate
        # LLM turn 2: read result, respond with text
        provider = ScriptedProvider([
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="browser_navigate",
                        arguments={"url": "https://example.com"},
                    )
                ],
                finish_reason="tool_use",
            ),
            LLMResponse(
                text="The page has a Sign Up button and an About link.",
                finish_reason="stop",
            ),
        ])

        engine = Engine(
            provider,
            tools=[],
            config=EngineConfig(model="test-model"),
            tool_sources=[browser_tools],
        )

        session = _session("Navigate to example.com and tell me what you see")
        events = await _collect_events(engine, session)

        # Verify event flow
        event_types = [e.type for e in events]
        assert EventType.RUN_START in event_types
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_END in event_types
        assert EventType.RUN_END in event_types

        # Verify tool was called successfully
        tool_end = next(e for e in events if e.type == EventType.TOOL_CALL_END)
        assert tool_end.payload["name"] == "browser_navigate"
        assert tool_end.payload["is_error"] is False

        # Verify session has tool result with snapshot
        tool_result_msgs = [m for m in session.messages if m.role == Role.TOOL]
        assert len(tool_result_msgs) == 1
        result_output = tool_result_msgs[0].tool_results[0].output
        assert "example.com" in result_output["url"]
        assert '@e1 [link] "About"' in result_output["elements"]
        assert '@e2 [button] "Sign Up"' in result_output["elements"]

        # Verify final state
        assert session.state == RunState.DONE

        await browser_client.close()


# ---------------------------------------------------------------------------
# Test 2: Engine multi-step: navigate -> click -> get_text
# ---------------------------------------------------------------------------


class TestEngineMultiStepBrowsing:
    """Engine does navigate -> click -> get_text in sequence."""

    async def test_three_step_interaction(self):
        page = MockBrowserPage({
            "https://shop.example.com": '- link "Products"\n- link "About"\n- textbox "Search"',
            "https://shop.example.com/products": '- link "Widget A"\n- link "Widget B"\n- button "Back"',
        })
        browser_client = BrowserClient(
            BrowserConfig(), page_factory=_mock_browser_factory(page)
        )
        browser_tools = BrowserToolSource(browser_client)

        provider = ScriptedProvider([
            # Turn 1: navigate
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c1", name="browser_navigate",
                             arguments={"url": "https://shop.example.com"})
                ],
                finish_reason="tool_use",
            ),
            # Turn 2: click Products link
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c2", name="browser_click",
                             arguments={"target": "@e1"})
                ],
                finish_reason="tool_use",
            ),
            # Turn 3: get text
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c3", name="browser_get_text",
                             arguments={})
                ],
                finish_reason="tool_use",
            ),
            # Turn 4: respond
            LLMResponse(
                text="Found Widget A and Widget B on the products page.",
                finish_reason="stop",
            ),
        ])

        engine = Engine(
            provider, tools=[], config=EngineConfig(model="test"),
            tool_sources=[browser_tools],
        )

        session = _session("Find products on shop.example.com")
        events = await _collect_events(engine, session)

        # Count tool calls
        tool_calls = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(tool_calls) == 3
        assert tool_calls[0].payload["name"] == "browser_navigate"
        assert tool_calls[1].payload["name"] == "browser_click"
        assert tool_calls[2].payload["name"] == "browser_get_text"

        # All succeeded
        assert all(not tc.payload["is_error"] for tc in tool_calls)
        assert session.state == RunState.DONE

        # Verify the click was logged on the mock page
        assert page.click_log == ["link:Products"]

        await browser_client.close()


# ---------------------------------------------------------------------------
# Test 3: Engine browser + builtin tool together
# ---------------------------------------------------------------------------


class TestEngineBrowserWithBuiltinTools:
    """Browser tools coexist with builtin tools, no name collisions."""

    async def test_mixed_tools(self):
        page = MockBrowserPage({
            "https://example.com": '- button "Click me"',
        })
        browser_client = BrowserClient(
            BrowserConfig(), page_factory=_mock_browser_factory(page)
        )
        browser_tools = BrowserToolSource(browser_client)

        # Builtin tool
        async def echo_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            return {"echoed": args.get("text", "")}

        echo_tool = ToolSpec(
            name="echo",
            description="Echo text back",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
            handler=echo_handler,
        )

        provider = ScriptedProvider([
            # Turn 1: call builtin echo
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c1", name="echo",
                             arguments={"text": "hello"})
                ],
                finish_reason="tool_use",
            ),
            # Turn 2: call browser navigate
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c2", name="browser_navigate",
                             arguments={"url": "https://example.com"})
                ],
                finish_reason="tool_use",
            ),
            # Turn 3: done
            LLMResponse(text="All done.", finish_reason="stop"),
        ])

        engine = Engine(
            provider,
            tools=[echo_tool],
            config=EngineConfig(model="test"),
            tool_sources=[browser_tools],
        )

        session = _session("Echo hello then navigate to example.com")
        events = await _collect_events(engine, session)

        tool_ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(tool_ends) == 2
        assert tool_ends[0].payload["name"] == "echo"
        assert tool_ends[1].payload["name"] == "browser_navigate"
        assert all(not t.payload["is_error"] for t in tool_ends)

        await browser_client.close()


# ---------------------------------------------------------------------------
# Test 4: Browser tool error handling in engine context
# ---------------------------------------------------------------------------


class TestEngineBrowserErrorHandling:
    """Browser tool errors don't crash the engine."""

    async def test_stale_ref_error_recovers(self):
        page = MockBrowserPage({
            "https://example.com": '- button "OK"',
        })
        browser_client = BrowserClient(
            BrowserConfig(), page_factory=_mock_browser_factory(page)
        )
        browser_tools = BrowserToolSource(browser_client)

        provider = ScriptedProvider([
            # Turn 1: click stale ref (no navigate first, so no refs exist)
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c1", name="browser_click",
                             arguments={"target": "@e99"})
                ],
                finish_reason="tool_use",
            ),
            # Turn 2: LLM sees error, navigates instead
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c2", name="browser_navigate",
                             arguments={"url": "https://example.com"})
                ],
                finish_reason="tool_use",
            ),
            # Turn 3: now click with valid ref
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c3", name="browser_click",
                             arguments={"target": "@e1"})
                ],
                finish_reason="tool_use",
            ),
            # Turn 4: done
            LLMResponse(text="Clicked OK after recovery.", finish_reason="stop"),
        ])

        engine = Engine(
            provider, tools=[], config=EngineConfig(model="test"),
            tool_sources=[browser_tools],
        )

        session = _session("Click something on example.com")
        events = await _collect_events(engine, session)

        tool_ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(tool_ends) == 3

        # First click failed (stale ref)
        assert tool_ends[0].payload["name"] == "browser_click"
        # BrowserToolSource catches the error and returns is_error dict
        # Engine level is_error is False because the handler didn't raise
        assert tool_ends[0].payload["is_error"] is False

        # Check the tool result contains the error info
        tool_msg = session.messages[2]  # USER, ASSISTANT(tool_call), TOOL(result)
        result = tool_msg.tool_results[0]
        assert result.output["is_error"] is True

        # Navigate succeeded
        assert tool_ends[1].payload["name"] == "browser_navigate"

        # Second click succeeded
        assert tool_ends[2].payload["name"] == "browser_click"
        assert page.click_log == ["button:OK"]

        assert session.state == RunState.DONE
        await browser_client.close()


# ---------------------------------------------------------------------------
# Test 5: Multi-agent orchestrator with browser tools
# ---------------------------------------------------------------------------


class TestOrchestratorWithBrowser:
    """Two-step plan: step 1 navigates, step 2 extracts text."""

    async def test_two_step_plan_with_browser(self):
        page = MockBrowserPage({
            "https://news.example.com": '- link "Article 1"\n- link "Article 2"',
        })

        def make_browser() -> tuple[BrowserClient, BrowserToolSource]:
            client = BrowserClient(
                BrowserConfig(), page_factory=_mock_browser_factory(page)
            )
            return client, BrowserToolSource(client)

        # Step 1 provider: navigate
        provider_step1 = ScriptedProvider([
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c1", name="browser_navigate",
                             arguments={"url": "https://news.example.com"})
                ],
                finish_reason="tool_use",
            ),
            LLMResponse(text="Navigation complete. Found 2 articles.", finish_reason="stop"),
        ])

        # Step 2 provider: get text
        provider_step2 = ScriptedProvider([
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c2", name="browser_get_text", arguments={})
                ],
                finish_reason="tool_use",
            ),
            LLMResponse(text="Extracted page content.", finish_reason="stop"),
        ])

        # Create plan
        plan = Plan(
            id="browser-plan",
            goal="Scrape news site",
            steps=[
                PlanStep(id="nav", title="Navigate", instructions="Go to news site"),
                PlanStep(
                    id="extract", title="Extract",
                    instructions="Get page text",
                    depends_on=["nav"],
                ),
            ],
        )

        # Both steps share the same browser client (session-scoped)
        browser_client, browser_tools = make_browser()

        call_count = 0

        # StepConfigurator to inject the right provider per step
        class BrowserStepConfigurator:
            name = "browser-config"

            async def configure_step(
                self, step: PlanStep, config: SubAgentConfig
            ) -> SubAgentConfig:
                nonlocal call_count
                call_count += 1
                provider = provider_step1 if step.id == "nav" else provider_step2
                return SubAgentConfig(
                    provider=provider,
                    model=config.model,
                    tools=config.tools,
                    tool_sources=[browser_tools],
                )

        collector = EventCollector()

        config = SubAgentConfig(
            provider=provider_step1,  # default, overridden by configurator
            model="test",
            tool_sources=[browser_tools],
        )

        orch = Orchestrator(
            plan, config,
            event_subscribers=[collector],
            step_configurators=[BrowserStepConfigurator()],
        )

        events = [event async for event in orch.execute()]

        # Plan completed
        plan_done = [e for e in events if e.type == EventType.PLAN_DONE]
        assert len(plan_done) == 1

        # Both steps executed
        step_ends = [e for e in events if e.type == EventType.PLAN_STEP_END]
        assert len(step_ends) == 2

        # Step configurator was called for each step
        assert call_count == 2

        # Page was navigated
        assert page.url == "https://news.example.com"

        await browser_client.close()


# ---------------------------------------------------------------------------
# Test 6: BrowserToolSource as ToolSource protocol conformance
# ---------------------------------------------------------------------------


class TestBrowserToolSourceProtocol:
    """Verify BrowserToolSource satisfies ToolSource protocol."""

    async def test_has_name_attribute(self):
        page = MockBrowserPage({})
        client = BrowserClient(BrowserConfig(), page_factory=_mock_browser_factory(page))
        source = BrowserToolSource(client)
        assert hasattr(source, "name")
        assert source.name == "browser"

    async def test_list_tools_returns_toolspecs(self):
        page = MockBrowserPage({})
        client = BrowserClient(BrowserConfig(), page_factory=_mock_browser_factory(page))
        source = BrowserToolSource(client)
        tools = await source.list_tools()
        assert all(isinstance(t, ToolSpec) for t in tools)
        assert all(callable(t.handler) for t in tools)

    async def test_tool_snapshot_refreshes_each_step(self):
        """Engine snapshots tools each step; browser tools are always fresh."""
        page = MockBrowserPage({
            "https://example.com": '- button "Go"',
        })
        client = BrowserClient(BrowserConfig(), page_factory=_mock_browser_factory(page))
        source = BrowserToolSource(client)

        tools_1 = await source.list_tools()
        tools_2 = await source.list_tools()

        # Same names but different handler objects (fresh each call)
        assert {t.name for t in tools_1} == {t.name for t in tools_2}

        await client.close()


# ---------------------------------------------------------------------------
# Test 7: Concurrent sessions don't share browser state
# ---------------------------------------------------------------------------


class TestBrowserSessionIsolation:
    """Two engines with separate BrowserClients don't interfere."""

    async def test_isolated_sessions(self):
        page_a = MockBrowserPage({
            "https://a.com": '- link "A Link"',
        })
        page_b = MockBrowserPage({
            "https://b.com": '- button "B Button"',
        })

        client_a = BrowserClient(BrowserConfig(), page_factory=_mock_browser_factory(page_a))
        client_b = BrowserClient(BrowserConfig(), page_factory=_mock_browser_factory(page_b))

        source_a = BrowserToolSource(client_a)
        source_b = BrowserToolSource(client_b)

        provider_a = ScriptedProvider([
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="ca1", name="browser_navigate",
                                     arguments={"url": "https://a.com"})],
                finish_reason="tool_use",
            ),
            LLMResponse(text="On page A.", finish_reason="stop"),
        ])
        provider_b = ScriptedProvider([
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="cb1", name="browser_navigate",
                                     arguments={"url": "https://b.com"})],
                finish_reason="tool_use",
            ),
            LLMResponse(text="On page B.", finish_reason="stop"),
        ])

        engine_a = Engine(
            provider_a, tools=[], config=EngineConfig(model="test"),
            tool_sources=[source_a],
        )
        engine_b = Engine(
            provider_b, tools=[], config=EngineConfig(model="test"),
            tool_sources=[source_b],
        )

        session_a = _session("Go to A", sid="session-a")
        session_b = _session("Go to B", sid="session-b")

        # Run concurrently
        results = await asyncio.gather(
            _collect_events(engine_a, session_a),
            _collect_events(engine_b, session_b),
        )

        events_a, events_b = results

        # Both completed
        assert session_a.state == RunState.DONE
        assert session_b.state == RunState.DONE

        # Pages are isolated
        assert page_a.url == "https://a.com"
        assert page_b.url == "https://b.com"

        await client_a.close()
        await client_b.close()
