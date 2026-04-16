from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.browser.client import BrowserClient
from topsport_agent.browser.tools import BrowserToolSource
from topsport_agent.browser.types import BrowserConfig, PageSnapshot, SnapshotEntry
from topsport_agent.browser.snapshot import (
    INTERACTIVE_ROLES,
    _parse_aria_yaml,
    build_ref_map,
    take_snapshot,
)
from topsport_agent.types.tool import ToolContext, ToolSpec


class TestBrowserConfig:
    def test_defaults(self):
        cfg = BrowserConfig()
        assert cfg.headless is True
        assert cfg.viewport_width == 1280
        assert cfg.viewport_height == 720
        assert cfg.default_timeout == 30.0

    def test_custom(self):
        cfg = BrowserConfig(headless=False, viewport_width=800, viewport_height=600, default_timeout=10.0)
        assert cfg.headless is False
        assert cfg.viewport_width == 800


class TestSnapshotEntry:
    def test_fields(self):
        e = SnapshotEntry(ref="@e1", role="button", name="Submit")
        assert e.ref == "@e1"
        assert e.role == "button"
        assert e.name == "Submit"
        assert e.tag == ""
        assert e.attributes == {}


class TestPageSnapshot:
    def test_to_text_empty(self):
        snap = PageSnapshot(url="https://example.com", title="Example")
        text = snap.to_text()
        assert "URL: https://example.com" in text
        assert "Title: Example" in text

    def test_to_text_with_entries(self):
        snap = PageSnapshot(
            url="https://example.com",
            title="Example",
            entries=[
                SnapshotEntry(ref="@e1", role="textbox", name="Email"),
                SnapshotEntry(ref="@e2", role="button", name="Submit"),
            ],
        )
        text = snap.to_text()
        assert '@e1 [textbox] "Email"' in text
        assert '@e2 [button] "Submit"' in text

    def test_to_text_entry_without_name(self):
        snap = PageSnapshot(
            url="https://x.com",
            title="X",
            entries=[SnapshotEntry(ref="@e1", role="link", name="")],
        )
        text = snap.to_text()
        assert "@e1 [link]" in text
        assert '""' not in text


def _make_aria_yaml(*lines: str) -> str:
    """Helper: build aria_snapshot YAML from shorthand lines."""
    return "\n".join(lines)


class MockAriaLocator:
    """Mock locator returned by page.locator('body') for aria_snapshot and text_content."""

    def __init__(self, yaml_text: str) -> None:
        self._yaml = yaml_text

    async def aria_snapshot(self) -> str:
        return self._yaml

    async def text_content(self) -> str:
        return "Full page text content"

    async def count(self) -> int:
        return 1

    @property
    def first(self) -> MockAriaLocator:
        return self


class MockPage:
    """Mock Playwright Page for testing BrowserClient without Playwright."""

    def __init__(
        self,
        *,
        url: str = "https://example.com",
        page_title: str = "Example",
        aria_yaml: str = "",
    ) -> None:
        self.url = url
        self._title = page_title
        self._aria_yaml = aria_yaml
        self.click_log: list[str] = []
        self.fill_log: list[tuple[str, str]] = []

    async def title(self) -> str:
        return self._title

    async def goto(self, url: str, **kwargs: Any) -> None:
        self.url = url

    def locator(self, selector: str) -> MockAriaLocator | MockInteractionLocator:
        if selector == "body":
            return MockAriaLocator(self._aria_yaml)
        return MockInteractionLocator(self, "css", selector)

    def get_by_role(self, role: str, *, name: str = "") -> MockInteractionLocator:
        return MockInteractionLocator(self, role, name)

    async def screenshot(self, *, path: str = "", **kwargs: Any) -> bytes:
        if path:
            Path(path).write_bytes(b"fake-png-data")
        return b"fake-png-data"

    async def text_content(self) -> str:
        return "Full page text content"


class MockInteractionLocator:
    def __init__(self, page: MockPage, role: str, name: str) -> None:
        self._page = page
        self._role = role
        self._name = name
        self._mock_count = 0  # default: element not found (for content selectors)

    async def click(self, **kwargs: Any) -> None:
        self._page.click_log.append(f"{self._role}:{self._name}")

    async def fill(self, text: str, **kwargs: Any) -> None:
        self._page.fill_log.append((f"{self._role}:{self._name}", text))

    async def text_content(self) -> str:
        return f"text of {self._role}:{self._name}"

    @property
    def first(self) -> MockInteractionLocator:
        return self

    async def count(self) -> int:
        return self._mock_count


def _mock_page_factory(page: MockPage):
    @contextlib.asynccontextmanager
    async def factory():
        yield page

    return factory


# --- Snapshot Tests ---


class TestParseAriaYaml:
    def test_extracts_interactive_elements(self):
        yaml = _make_aria_yaml(
            '- textbox "Email"',
            '- textbox "Password"',
            '- button "Login"',
        )
        entries = _parse_aria_yaml(yaml)
        assert len(entries) == 3
        assert entries[0].ref == "@e1"
        assert entries[0].role == "textbox"
        assert entries[0].name == "Email"
        assert entries[2].ref == "@e3"
        assert entries[2].role == "button"

    def test_ignores_non_interactive_roles(self):
        yaml = _make_aria_yaml(
            '- heading "Welcome" [level=1]',
            '- paragraph: some text',
            '- button "Click me"',
        )
        entries = _parse_aria_yaml(yaml)
        assert len(entries) == 1
        assert entries[0].role == "button"

    def test_nested_children(self):
        yaml = _make_aria_yaml(
            '- navigation "nav":',
            '  - link "Home"',
            '  - link "About"',
        )
        entries = _parse_aria_yaml(yaml)
        assert len(entries) == 2
        assert entries[0].ref == "@e1"
        assert entries[0].name == "Home"
        assert entries[1].ref == "@e2"

    def test_empty_string(self):
        entries = _parse_aria_yaml("")
        assert entries == []

    def test_sequential_numbering_skips_non_interactive(self):
        yaml = _make_aria_yaml(
            '- link "A"',
            '- heading "skip" [level=2]',
            '- link "B"',
            '- textbox "C"',
        )
        entries = _parse_aria_yaml(yaml)
        refs = [e.ref for e in entries]
        assert refs == ["@e1", "@e2", "@e3"]

    def test_element_without_name(self):
        yaml = _make_aria_yaml('- button')
        entries = _parse_aria_yaml(yaml)
        assert len(entries) == 1
        assert entries[0].name == ""

    def test_element_with_attributes(self):
        yaml = _make_aria_yaml('- checkbox "Agree" [checked]')
        entries = _parse_aria_yaml(yaml)
        assert len(entries) == 1
        assert entries[0].role == "checkbox"
        assert entries[0].name == "Agree"


class TestTakeSnapshot:
    async def test_extracts_from_page(self):
        yaml = _make_aria_yaml(
            '- textbox "Email"',
            '- button "Submit"',
        )
        page = MockPage(aria_yaml=yaml)
        snap = await take_snapshot(page)
        assert len(snap.entries) == 2
        assert snap.entries[0].ref == "@e1"
        assert snap.url == "https://example.com"

    async def test_empty_page(self):
        page = MockPage(aria_yaml="")
        snap = await take_snapshot(page)
        assert snap.entries == []
        assert snap.url == "https://example.com"


class TestBuildRefMap:
    def test_builds_map(self):
        snap = PageSnapshot(
            url="https://example.com",
            title="Test",
            entries=[
                SnapshotEntry(ref="@e1", role="button", name="Submit"),
                SnapshotEntry(ref="@e2", role="textbox", name="Email"),
            ],
        )
        ref_map = build_ref_map(snap)
        assert ref_map == {
            "@e1": ("button", "Submit"),
            "@e2": ("textbox", "Email"),
        }


# --- Client Tests ---


class TestBrowserClient:
    async def test_lazy_init_no_page_until_first_call(self):
        created = []

        @contextlib.asynccontextmanager
        async def factory():
            page = MockPage()
            created.append(page)
            yield page

        client = BrowserClient(BrowserConfig(), page_factory=factory)
        assert len(created) == 0
        await client.navigate("https://example.com")
        assert len(created) == 1

    async def test_navigate_returns_snapshot(self):
        yaml = _make_aria_yaml(
            '- button "Go"',
            '- textbox "Search"',
        )
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))

        snap = await client.navigate("https://example.com")
        assert snap.url == "https://example.com"
        assert len(snap.entries) == 2
        assert snap.entries[0].ref == "@e1"
        assert page.url == "https://example.com"

    async def test_click_with_ref(self):
        yaml = _make_aria_yaml('- button "Submit"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))

        await client.navigate("https://example.com")
        result = await client.click("@e1")
        assert page.click_log == ["button:Submit"]
        assert "clicked" in result

    async def test_click_with_css_selector(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")

        result = await client.click("#my-button")
        assert page.click_log == ["css:#my-button"]

    async def test_stale_ref_raises_error(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))

        with pytest.raises(KeyError, match="@e99"):
            await client.click("@e99")

    async def test_type_text(self):
        yaml = _make_aria_yaml('- textbox "Email"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))

        await client.navigate("https://example.com")
        await client.type_text("@e1", "user@example.com")
        assert page.fill_log == [("textbox:Email", "user@example.com")]

    async def test_screenshot_returns_path(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "shot.png")
            result = await client.screenshot(path)
            assert result == path
            assert Path(path).exists()

    async def test_screenshot_auto_path(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")

        result = await client.screenshot()
        assert result.endswith(".png")

    async def test_get_text_full_page(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")

        text = await client.get_text()
        assert "Full page text content" in text

    async def test_get_text_with_ref(self):
        yaml = _make_aria_yaml('- link "Home"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")

        text = await client.get_text("@e1")
        assert "link:Home" in text

    async def test_close_releases_resources(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")
        await client.close()
        assert client._page is None

    async def test_navigate_without_factory_raises(self):
        client = BrowserClient(BrowserConfig())
        with pytest.raises(RuntimeError, match="page_factory"):
            await client.navigate("https://example.com")


# --- ToolSource Tests ---


@pytest.fixture
def cancel_event() -> asyncio.Event:
    return asyncio.Event()


def _tool_ctx(cancel_event: asyncio.Event) -> ToolContext:
    return ToolContext(session_id="test", call_id="c1", cancel_event=cancel_event)


class TestBrowserToolSource:
    async def test_list_tools_returns_six(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        tools = await source.list_tools()
        assert len(tools) == 6
        names = {t.name for t in tools}
        assert names == {
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "browser_screenshot",
            "browser_get_text",
        }

    async def test_all_tools_are_valid_toolspecs(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        for tool in await source.list_tools():
            assert isinstance(tool, ToolSpec)
            assert tool.name
            assert tool.description
            assert tool.parameters["type"] == "object"
            assert callable(tool.handler)

    async def test_navigate_handler(self, cancel_event: asyncio.Event):
        yaml = _make_aria_yaml('- button "Go"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        tools = await source.list_tools()
        nav = next(t for t in tools if t.name == "browser_navigate")
        result = await nav.handler({"url": "https://example.com"}, _tool_ctx(cancel_event))
        assert "URL: https://example.com" in result["elements"]
        assert '@e1 [button] "Go"' in result["elements"]

    async def test_click_handler(self, cancel_event: asyncio.Event):
        yaml = _make_aria_yaml('- button "OK"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        click = next(t for t in tools if t.name == "browser_click")
        result = await click.handler({"target": "@e1"}, _tool_ctx(cancel_event))
        assert result["clicked"] == "@e1"

    async def test_type_handler(self, cancel_event: asyncio.Event):
        yaml = _make_aria_yaml('- textbox "Name"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        typ = next(t for t in tools if t.name == "browser_type")
        result = await typ.handler({"target": "@e1", "text": "hello"}, _tool_ctx(cancel_event))
        assert result["typed"] == "hello"

    async def test_screenshot_handler(self, cancel_event: asyncio.Event):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        ss = next(t for t in tools if t.name == "browser_screenshot")
        result = await ss.handler({}, _tool_ctx(cancel_event))
        assert "path" in result
        assert result["path"].endswith(".png")

    async def test_get_text_handler(self, cancel_event: asyncio.Event):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        gt = next(t for t in tools if t.name == "browser_get_text")
        result = await gt.handler({}, _tool_ctx(cancel_event))
        assert "text" in result

    async def test_handler_catches_errors(self, cancel_event: asyncio.Event):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        tools = await source.list_tools()
        click = next(t for t in tools if t.name == "browser_click")
        result = await click.handler({"target": "@e99"}, _tool_ctx(cancel_event))
        assert result["is_error"] is True
        assert "error" in result

    async def test_source_name(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)
        assert source.name == "browser"


class TestBrowserPublicAPI:
    def test_public_imports(self):
        from topsport_agent.browser import (
            BrowserClient,
            BrowserConfig,
            BrowserToolSource,
            PageFactory,
        )
        assert BrowserClient is not None
        assert BrowserConfig is not None
        assert BrowserToolSource is not None
        assert PageFactory is not None
