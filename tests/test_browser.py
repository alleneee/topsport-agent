from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.browser.client import BrowserClient
from topsport_agent.browser.snapshot import (
    _parse_aria_yaml,
    build_ref_map,
    take_snapshot,
)
from topsport_agent.browser.tools import BrowserToolSource
from topsport_agent.browser.types import BrowserConfig, PageSnapshot, SnapshotEntry
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


class MockKeyboard:
    def __init__(self, page: MockPage) -> None:
        self._page = page

    async def press(self, key: str) -> None:
        self._page.press_log.append(("page", key))


class MockPage:
    """Mock Playwright Page for testing BrowserClient without Playwright."""

    def __init__(
        self,
        *,
        url: str = "https://example.com",
        page_title: str = "Example",
        aria_yaml: str = "",
        frame_yamls: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self._title = page_title
        self._aria_yaml = aria_yaml
        self._frame_yamls = frame_yamls or {}
        self.click_log: list[str] = []
        self.fill_log: list[tuple[str, str]] = []
        self.press_log: list[tuple[str, str]] = []
        self.select_log: list[tuple[str, list[str]]] = []
        self.wait_log: list[tuple[str, str]] = []
        self._history: list[str] = []
        self.keyboard = MockKeyboard(self)

    async def title(self) -> str:
        return self._title

    async def goto(self, url: str, **kwargs: Any) -> None:
        if self.url and self.url != url:
            self._history.append(self.url)
        self.url = url

    async def go_back(self) -> None:
        if self._history:
            self.url = self._history.pop()

    def locator(self, selector: str) -> MockAriaLocator | MockInteractionLocator:
        if selector == "body":
            return MockAriaLocator(self._aria_yaml)
        return MockInteractionLocator(self, "css", selector)

    def get_by_role(self, role: str, *, name: str = "") -> MockInteractionLocator:
        return MockInteractionLocator(self, role, name)

    def frame_locator(self, selector: str) -> MockFrameLocator:
        return MockFrameLocator(self, selector)

    async def screenshot(self, *, path: str = "", **kwargs: Any) -> bytes:
        if path:
            Path(path).write_bytes(b"fake-png-data")
        return b"fake-png-data"

    async def text_content(self) -> str:
        return "Full page text content"


class MockFrameLocator:
    """Mock of Playwright FrameLocator: proxies locator/get_by_role to the page."""

    def __init__(self, page: MockPage, frame_selector: str) -> None:
        self._page = page
        self._frame = frame_selector
        self._aria_yaml = page._frame_yamls.get(frame_selector, "")

    def locator(self, selector: str) -> MockAriaLocator | MockInteractionLocator:
        if selector == "body":
            return MockAriaLocator(self._aria_yaml)
        return MockInteractionLocator(self._page, "css", selector, frame=self._frame)

    def get_by_role(self, role: str, *, name: str = "") -> MockInteractionLocator:
        return MockInteractionLocator(self._page, role, name, frame=self._frame)


class MockInteractionLocator:
    def __init__(
        self,
        page: MockPage,
        role: str,
        name: str,
        *,
        frame: str = "",
        nth: int | None = None,
    ) -> None:
        self._page = page
        self._role = role
        self._name = name
        self._frame = frame
        self._nth = nth
        self._mock_count = 0  # default: element not found (for content selectors)

    def _tag(self) -> str:
        frame_part = f"[{self._frame}]" if self._frame else ""
        nth_part = f"#{self._nth}" if self._nth is not None else ""
        return f"{frame_part}{self._role}:{self._name}{nth_part}"

    async def click(self, **kwargs: Any) -> None:
        self._page.click_log.append(self._tag())

    async def fill(self, text: str, **kwargs: Any) -> None:
        self._page.fill_log.append((self._tag(), text))

    async def press(self, key: str, **kwargs: Any) -> None:
        self._page.press_log.append((self._tag(), key))

    async def select_option(self, values: Any, **kwargs: Any) -> list[str]:
        out = [values] if isinstance(values, str) else list(values)
        self._page.select_log.append((self._tag(), out))
        return out

    async def wait_for(self, *, state: str = "visible", **kwargs: Any) -> None:
        self._page.wait_log.append((self._tag(), state))

    async def text_content(self) -> str:
        return f"text of {self._role}:{self._name}"

    @property
    def first(self) -> MockInteractionLocator:
        return self

    def nth(self, n: int) -> MockInteractionLocator:
        return MockInteractionLocator(self._page, self._role, self._name, frame=self._frame, nth=n)

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

    def test_inline_ref_and_nth_attributes_still_parse(self):
        # QwenPaw-style enriched aria output with embedded [ref=...] / [nth=...]
        yaml = _make_aria_yaml(
            '- button "Save" [ref=e1] [nth=0]',
            '- button "Save" [ref=e2] [nth=1]:',
        )
        entries = _parse_aria_yaml(yaml)
        assert len(entries) == 2
        assert entries[0].role == "button"
        assert entries[0].name == "Save"

    def test_closing_slash_role_ignored(self):
        yaml = _make_aria_yaml(
            '- button "Go"',
            '- /button',
        )
        entries = _parse_aria_yaml(yaml)
        assert len(entries) == 1
        assert entries[0].name == "Go"

    def test_duplicates_get_sequential_nth(self):
        yaml = _make_aria_yaml(
            '- button "Save"',
            '- textbox "Name"',
            '- button "Save"',
            '- button "Save"',
        )
        entries = _parse_aria_yaml(yaml)
        by_ref = {e.ref: e for e in entries}
        assert by_ref["@e1"].role == "button" and by_ref["@e1"].nth == 0
        assert by_ref["@e2"].role == "textbox" and by_ref["@e2"].nth is None
        assert by_ref["@e3"].nth == 1
        assert by_ref["@e4"].nth == 2

    def test_unique_entry_has_no_nth(self):
        yaml = _make_aria_yaml('- button "Only"')
        entries = _parse_aria_yaml(yaml)
        assert entries[0].nth is None


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
            "@e1": ("button", "Submit", None),
            "@e2": ("textbox", "Email", None),
        }

    def test_preserves_nth_for_duplicates(self):
        snap = PageSnapshot(
            url="https://x",
            title="X",
            entries=[
                SnapshotEntry(ref="@e1", role="button", name="Save", nth=0),
                SnapshotEntry(ref="@e2", role="button", name="Save", nth=1),
            ],
        )
        assert build_ref_map(snap) == {
            "@e1": ("button", "Save", 0),
            "@e2": ("button", "Save", 1),
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

    async def test_click_duplicate_uses_nth(self):
        # 同 role + name 重复 → ref 带上 nth，click 必须走 .nth(n) 分支
        yaml = _make_aria_yaml(
            '- button "Save"',
            '- button "Save"',
            '- button "Save"',
        )
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")
        await client.click("@e2")  # 第二个 "Save"
        assert page.click_log == ["button:Save#1"]

    async def test_click_unique_ref_still_uses_first(self):
        yaml = _make_aria_yaml('- button "Only"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")
        await client.click("@e1")
        # nth 为 None 时不应带 "#n"
        assert page.click_log == ["button:Only"]

    async def test_snapshot_with_frame_selector_scopes_refs(self):
        frame_yaml = _make_aria_yaml('- button "Signup"')
        page = MockPage(aria_yaml="", frame_yamls={"iframe#mc": frame_yaml})
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")

        snap = await client.snapshot(frame_selector="iframe#mc")
        assert snap.frame_selector == "iframe#mc"
        assert [e.name for e in snap.entries] == ["Signup"]

    async def test_click_inherits_frame_from_snapshot(self):
        frame_yaml = _make_aria_yaml('- button "Signup"')
        page = MockPage(aria_yaml="", frame_yamls={"iframe#mc": frame_yaml})
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://example.com")
        await client.snapshot(frame_selector="iframe#mc")
        await client.click("@e1")
        assert page.click_log == ["[iframe#mc]button:Signup"]

    async def test_navigate_clears_frame_scope(self):
        frame_yaml = _make_aria_yaml('- button "Signup"')
        page = MockPage(aria_yaml="", frame_yamls={"iframe#mc": frame_yaml})
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://a")
        await client.snapshot(frame_selector="iframe#mc")
        assert client._snapshot_frame == "iframe#mc"
        await client.navigate("https://b")
        assert client._snapshot_frame == ""

    async def test_navigate_back_restores_previous_url(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://a")
        await client.navigate("https://b")
        snap = await client.navigate_back()
        assert snap.url == "https://a"

    async def test_press_key_page_level(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://x")
        await client.press_key("Escape")
        assert page.press_log == [("page", "Escape")]

    async def test_press_key_with_target(self):
        yaml = _make_aria_yaml('- textbox "Q"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://x")
        await client.press_key("Enter", target="@e1")
        assert page.press_log == [("textbox:Q", "Enter")]

    async def test_select_option_returns_selected(self):
        yaml = _make_aria_yaml('- combobox "Country"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://x")
        selected = await client.select_option("@e1", ["cn", "jp"])
        assert selected == ["cn", "jp"]
        assert page.select_log == [("combobox:Country", ["cn", "jp"])]

    async def test_wait_for_requires_selector_or_seconds(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://x")
        with pytest.raises(ValueError, match="selector"):
            await client.wait_for()

    async def test_wait_for_selector(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://x")
        await client.wait_for(selector=".banner", state="hidden")
        assert page.wait_log == [("css:.banner", "hidden")]

    async def test_wait_for_seconds_sleeps(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        await client.navigate("https://x")
        # 非常短的 sleep，主要验证不抛异常；selector 留空
        await client.wait_for(seconds=0.001)


# --- ToolSource Tests ---


@pytest.fixture
def cancel_event() -> asyncio.Event:
    return asyncio.Event()


def _tool_ctx(cancel_event: asyncio.Event) -> ToolContext:
    return ToolContext(session_id="test", call_id="c1", cancel_event=cancel_event)


class TestBrowserToolSource:
    async def test_list_tools_returns_full_set(self):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        tools = await source.list_tools()
        names = {t.name for t in tools}
        assert names == {
            "browser_navigate",
            "browser_back",
            "browser_snapshot",
            "browser_click",
            "browser_type",
            "browser_press_key",
            "browser_select_option",
            "browser_wait_for",
            "browser_screenshot",
            "browser_get_text",
        }
        assert len(tools) == len(names)

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

    async def test_back_handler(self, cancel_event: asyncio.Event):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://a")
        await client.navigate("https://b")
        tools = await source.list_tools()
        back = next(t for t in tools if t.name == "browser_back")
        result = await back.handler({}, _tool_ctx(cancel_event))
        assert result["url"] == "https://a"

    async def test_snapshot_handler_with_frame_selector(self, cancel_event: asyncio.Event):
        frame_yaml = _make_aria_yaml('- button "Signup"')
        page = MockPage(aria_yaml="", frame_yamls={"iframe#mc": frame_yaml})
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        snap = next(t for t in tools if t.name == "browser_snapshot")
        result = await snap.handler(
            {"frame_selector": "iframe#mc"}, _tool_ctx(cancel_event)
        )
        assert result["frame_selector"] == "iframe#mc"
        assert '@e1 [button] "Signup"' in result["elements"]

    async def test_press_key_handler(self, cancel_event: asyncio.Event):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        pk = next(t for t in tools if t.name == "browser_press_key")
        result = await pk.handler({"key": "Escape"}, _tool_ctx(cancel_event))
        assert result == {"pressed": "Escape", "target": "page"}
        assert page.press_log == [("page", "Escape")]

    async def test_select_option_handler_accepts_string(self, cancel_event: asyncio.Event):
        yaml = _make_aria_yaml('- combobox "C"')
        page = MockPage(aria_yaml=yaml)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        so = next(t for t in tools if t.name == "browser_select_option")
        result = await so.handler(
            {"target": "@e1", "values": "cn"}, _tool_ctx(cancel_event)
        )
        assert result["selected"] == ["cn"]

    async def test_wait_for_handler(self, cancel_event: asyncio.Event):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        wf = next(t for t in tools if t.name == "browser_wait_for")
        result = await wf.handler(
            {"selector": ".loader", "state": "hidden"}, _tool_ctx(cancel_event)
        )
        assert result == {"waited": True}
        assert page.wait_log == [("css:.loader", "hidden")]

    async def test_wait_for_handler_missing_args_returns_error(self, cancel_event: asyncio.Event):
        page = MockPage()
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        wf = next(t for t in tools if t.name == "browser_wait_for")
        result = await wf.handler({}, _tool_ctx(cancel_event))
        assert result["is_error"] is True


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


# ---------------------------------------------------------------------------
# H-S6 · URL policy: scheme + 内网 + metadata 拦截
# ---------------------------------------------------------------------------

from topsport_agent.browser import BrowserURLPolicy, BrowserURLRejected


class TestBrowserURLPolicy:
    def setup_method(self):
        self.policy = BrowserURLPolicy()

    def test_allows_https(self):
        self.policy.check("https://example.com/path")

    def test_allows_http(self):
        self.policy.check("http://example.com")

    def test_rejects_file_scheme(self):
        with pytest.raises(BrowserURLRejected, match="scheme"):
            self.policy.check("file:///etc/passwd")

    def test_rejects_javascript_scheme(self):
        with pytest.raises(BrowserURLRejected, match="scheme"):
            self.policy.check("javascript:alert(1)")

    def test_rejects_loopback_ipv4(self):
        with pytest.raises(BrowserURLRejected, match="loopback"):
            self.policy.check("http://127.0.0.1/admin")

    def test_rejects_ipv6_loopback(self):
        with pytest.raises(BrowserURLRejected, match="loopback"):
            self.policy.check("http://[::1]/")

    def test_rejects_private_rfc1918(self):
        with pytest.raises(BrowserURLRejected, match="non-public"):
            self.policy.check("http://10.0.0.1/")
        with pytest.raises(BrowserURLRejected, match="non-public"):
            self.policy.check("http://192.168.1.1/")
        with pytest.raises(BrowserURLRejected, match="non-public"):
            self.policy.check("http://172.16.0.1/")

    def test_rejects_link_local(self):
        with pytest.raises(BrowserURLRejected, match="non-public"):
            self.policy.check("http://169.254.1.1/")

    def test_rejects_aws_metadata(self):
        with pytest.raises(BrowserURLRejected, match="metadata"):
            self.policy.check("http://169.254.169.254/latest/meta-data/")

    def test_rejects_gcp_metadata(self):
        with pytest.raises(BrowserURLRejected, match="metadata"):
            self.policy.check("http://metadata.google.internal/")

    def test_rejects_aliyun_metadata(self):
        with pytest.raises(BrowserURLRejected, match="metadata"):
            self.policy.check("http://100.100.100.200/")

    def test_allow_private_still_blocks_metadata_and_loopback(self):
        policy = BrowserURLPolicy(allow_private=True)
        policy.check("http://10.0.0.1/")  # 内网通过
        with pytest.raises(BrowserURLRejected, match="loopback"):
            policy.check("http://127.0.0.1/")
        with pytest.raises(BrowserURLRejected, match="metadata"):
            policy.check("http://169.254.169.254/")

    def test_extra_host_denylist(self):
        policy = BrowserURLPolicy(extra_host_denylist=frozenset({"evil.example"}))
        with pytest.raises(BrowserURLRejected, match="operator denylist"):
            policy.check("https://evil.example/")


async def test_browser_client_navigate_rejects_policy_violation() -> None:
    """navigate 调用会在 page.goto 前触发策略检查。"""
    from topsport_agent.browser.client import BrowserClient
    from topsport_agent.browser.types import BrowserConfig

    page = MockPage()  # 初始 url = https://example.com
    initial_url = page.url
    client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))

    with pytest.raises(BrowserURLRejected):
        await client.navigate("file:///etc/passwd")
    # 策略在 _ensure_page 之前拒绝：page.goto 从未被调用，url 保持初始值
    assert page.url == initial_url
