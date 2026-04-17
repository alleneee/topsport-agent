# Browser Control Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a built-in browser control module that lets the agent navigate, interact with, and extract data from web pages using Playwright.

**Architecture:** `browser/` module follows the MCP module pattern: `types.py` (config dataclasses), `snapshot.py` (accessibility tree parsing + ref assignment), `client.py` (Playwright wrapper with injectable factory), `tools.py` (ToolSource providing 6 ToolSpecs). Playwright is an optional dependency group; all tests run with mock page objects.

**Tech Stack:** Playwright (async API), pytest-asyncio, existing ToolSource/ToolSpec protocols.

**Spec:** `docs/superpowers/specs/2026-04-16-browser-control-design.md`

---

## File Map

| File | Responsibility | Action |
|------|---------------|--------|
| `src/topsport_agent/browser/__init__.py` | Public exports | Create |
| `src/topsport_agent/browser/types.py` | BrowserConfig, SnapshotEntry, PageSnapshot | Create |
| `src/topsport_agent/browser/snapshot.py` | Accessibility tree walk, ref numbering, ref map | Create |
| `src/topsport_agent/browser/client.py` | BrowserClient: lazy Playwright lifecycle, page ops | Create |
| `src/topsport_agent/browser/tools.py` | BrowserToolSource(ToolSource): 6 ToolSpec definitions | Create |
| `tests/test_browser.py` | All browser module tests | Create |
| `pyproject.toml` | Add `browser` dependency group | Modify |

---

### Task 1: Types

**Files:**
- Create: `src/topsport_agent/browser/types.py`
- Create: `tests/test_browser.py`

- [ ] **Step 1: Write tests for types**

```python
# tests/test_browser.py
from __future__ import annotations

from topsport_agent.browser.types import BrowserConfig, PageSnapshot, SnapshotEntry


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_browser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topsport_agent.browser'`

- [ ] **Step 3: Implement types**

```python
# src/topsport_agent/browser/__init__.py
"""Built-in browser control module."""

# src/topsport_agent/browser/types.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BrowserConfig:
    """Browser launch configuration."""

    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    default_timeout: float = 30.0


@dataclass(slots=True)
class SnapshotEntry:
    """A single interactive element in the page snapshot."""

    ref: str
    role: str
    name: str
    tag: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PageSnapshot:
    """Structured snapshot of interactive elements on the current page."""

    url: str
    title: str
    entries: list[SnapshotEntry] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"URL: {self.url}", f"Title: {self.title}", ""]
        for e in self.entries:
            name_part = f' "{e.name}"' if e.name else ""
            lines.append(f"  {e.ref} [{e.role}]{name_part}")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_browser.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/browser/__init__.py src/topsport_agent/browser/types.py tests/test_browser.py
git commit -m "feat(browser): add types — BrowserConfig, SnapshotEntry, PageSnapshot"
```

---

### Task 2: Snapshot Engine

**Files:**
- Create: `src/topsport_agent/browser/snapshot.py`
- Modify: `tests/test_browser.py`

- [ ] **Step 1: Write tests for snapshot**

Append to `tests/test_browser.py`:

```python
from topsport_agent.browser.snapshot import (
    INTERACTIVE_ROLES,
    build_ref_map,
    take_snapshot,
)


def _make_tree(nodes: list[dict]) -> dict:
    """Helper: build a minimal accessibility tree dict."""
    return {"role": "WebArea", "name": "", "children": nodes}


class TestTakeSnapshot:
    async def test_extracts_interactive_elements(self):
        tree = _make_tree([
            {"role": "textbox", "name": "Email"},
            {"role": "textbox", "name": "Password"},
            {"role": "button", "name": "Login"},
        ])

        class FakePage:
            url = "https://example.com/login"
            async def title(self): return "Login"
            async def accessibility_snapshot(self): return tree

        snap = await take_snapshot(FakePage())
        assert len(snap.entries) == 3
        assert snap.entries[0].ref == "@e1"
        assert snap.entries[0].role == "textbox"
        assert snap.entries[0].name == "Email"
        assert snap.entries[2].ref == "@e3"
        assert snap.entries[2].role == "button"

    async def test_ignores_non_interactive_roles(self):
        tree = _make_tree([
            {"role": "heading", "name": "Welcome"},
            {"role": "text", "name": "some text"},
            {"role": "button", "name": "Click me"},
        ])

        class FakePage:
            url = "https://example.com"
            async def title(self): return "Test"
            async def accessibility_snapshot(self): return tree

        snap = await take_snapshot(FakePage())
        assert len(snap.entries) == 1
        assert snap.entries[0].role == "button"

    async def test_nested_children(self):
        tree = _make_tree([
            {
                "role": "navigation",
                "name": "nav",
                "children": [
                    {"role": "link", "name": "Home"},
                    {"role": "link", "name": "About"},
                ],
            },
        ])

        class FakePage:
            url = "https://example.com"
            async def title(self): return "Test"
            async def accessibility_snapshot(self): return tree

        snap = await take_snapshot(FakePage())
        assert len(snap.entries) == 2
        assert snap.entries[0].ref == "@e1"
        assert snap.entries[1].ref == "@e2"

    async def test_empty_tree(self):
        class FakePage:
            url = "https://example.com"
            async def title(self): return "Empty"
            async def accessibility_snapshot(self): return None

        snap = await take_snapshot(FakePage())
        assert snap.entries == []
        assert snap.url == "https://example.com"

    async def test_sequential_numbering(self):
        tree = _make_tree([
            {"role": "link", "name": "A"},
            {"role": "heading", "name": "skip"},
            {"role": "link", "name": "B"},
            {"role": "textbox", "name": "C"},
        ])

        class FakePage:
            url = "https://example.com"
            async def title(self): return "Test"
            async def accessibility_snapshot(self): return tree

        snap = await take_snapshot(FakePage())
        refs = [e.ref for e in snap.entries]
        assert refs == ["@e1", "@e2", "@e3"]


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_browser.py::TestTakeSnapshot -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topsport_agent.browser.snapshot'`

- [ ] **Step 3: Implement snapshot**

```python
# src/topsport_agent/browser/snapshot.py
"""Accessibility tree parsing and @ref assignment."""

from __future__ import annotations

from typing import Any

from .types import PageSnapshot, SnapshotEntry

INTERACTIVE_ROLES = frozenset({
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "menuitem",
    "tab",
    "switch",
    "searchbox",
    "slider",
    "spinbutton",
    "option",
    "menuitemcheckbox",
    "menuitemradio",
})


async def take_snapshot(page: Any) -> PageSnapshot:
    """Extract interactive elements from accessibility tree, assign sequential @refs."""
    tree = await page.accessibility_snapshot()
    url = page.url
    title = await page.title()

    entries: list[SnapshotEntry] = []
    counter = 0

    def walk(node: dict[str, Any]) -> None:
        nonlocal counter
        role = node.get("role", "")
        if role in INTERACTIVE_ROLES:
            counter += 1
            entries.append(
                SnapshotEntry(
                    ref=f"@e{counter}",
                    role=role,
                    name=node.get("name", ""),
                )
            )
        for child in node.get("children", []):
            walk(child)

    if tree:
        walk(tree)

    return PageSnapshot(url=url, title=title, entries=entries)


def build_ref_map(snapshot: PageSnapshot) -> dict[str, tuple[str, str]]:
    """Build {ref: (role, name)} mapping for locator resolution."""
    return {e.ref: (e.role, e.name) for e in snapshot.entries}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_browser.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/browser/snapshot.py tests/test_browser.py
git commit -m "feat(browser): add snapshot engine — accessibility tree parsing with @refs"
```

---

### Task 3: BrowserClient

**Files:**
- Create: `src/topsport_agent/browser/client.py`
- Modify: `tests/test_browser.py`

- [ ] **Step 1: Write mock page and factory helpers**

Add to the top of `tests/test_browser.py` (after existing imports):

```python
import asyncio
import contextlib
import tempfile
from pathlib import Path
from typing import Any

from topsport_agent.browser.client import BrowserClient
from topsport_agent.browser.snapshot import build_ref_map, take_snapshot


class MockPage:
    """Mock Playwright Page for testing BrowserClient without Playwright."""

    def __init__(
        self,
        *,
        url: str = "https://example.com",
        page_title: str = "Example",
        tree: dict | None = None,
    ) -> None:
        self.url = url
        self._title = page_title
        self._tree = tree or {"role": "WebArea", "name": "", "children": []}
        self.click_log: list[str] = []
        self.fill_log: list[tuple[str, str]] = []
        self._navigation_callback: Any = None
        self._closed = False

    async def title(self) -> str:
        return self._title

    async def accessibility_snapshot(self) -> dict | None:
        return self._tree

    async def goto(self, url: str, **kwargs: Any) -> None:
        self.url = url

    def get_by_role(self, role: str, *, name: str = "") -> MockLocator:
        return MockLocator(self, role, name)

    async def screenshot(self, *, path: str = "", **kwargs: Any) -> bytes:
        if path:
            Path(path).write_bytes(b"fake-png-data")
        return b"fake-png-data"

    async def text_content(self) -> str:
        return "Full page text content"

    def locator(self, selector: str) -> MockLocator:
        return MockLocator(self, "css", selector)


class MockLocator:
    def __init__(self, page: MockPage, role: str, name: str) -> None:
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
    def first(self) -> MockLocator:
        return self

    async def count(self) -> int:
        return 1


def _mock_page_factory(page: MockPage):
    @contextlib.asynccontextmanager
    async def factory():
        yield page

    return factory
```

- [ ] **Step 2: Write tests for BrowserClient**

Append to `tests/test_browser.py`:

```python
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
        tree = _make_tree([
            {"role": "button", "name": "Go"},
            {"role": "textbox", "name": "Search"},
        ])
        page = MockPage(tree=tree)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))

        snap = await client.navigate("https://example.com")
        assert snap.url == "https://example.com"
        assert len(snap.entries) == 2
        assert snap.entries[0].ref == "@e1"
        assert page.url == "https://example.com"

    async def test_click_with_ref(self):
        tree = _make_tree([{"role": "button", "name": "Submit"}])
        page = MockPage(tree=tree)
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
        tree = _make_tree([{"role": "textbox", "name": "Email"}])
        page = MockPage(tree=tree)
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
        tree = _make_tree([{"role": "link", "name": "Home"}])
        page = MockPage(tree=tree)
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_browser.py::TestBrowserClient -v`
Expected: FAIL — `ImportError: cannot import name 'BrowserClient'`

- [ ] **Step 4: Implement BrowserClient**

```python
# src/topsport_agent/browser/client.py
"""BrowserClient — Playwright wrapper with lazy init and injectable factory."""

from __future__ import annotations

import contextlib
import importlib
import tempfile
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any

from .snapshot import build_ref_map, take_snapshot
from .types import BrowserConfig, PageSnapshot

PageFactory = Callable[[], AbstractAsyncContextManager[Any]]


class BrowserClient:
    """Session-scoped browser control. Lazy-initialized on first operation.

    Production: use ``from_config()`` to build with real Playwright.
    Tests: pass a mock ``page_factory`` to ``__init__`` directly.
    """

    def __init__(
        self,
        config: BrowserConfig,
        page_factory: PageFactory | None = None,
    ) -> None:
        self._config = config
        self._page_factory = page_factory
        self._page: Any | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._ref_map: dict[str, tuple[str, str]] = {}

    @classmethod
    def from_config(cls, config: BrowserConfig) -> BrowserClient:
        """Production entry: creates real Playwright page_factory.

        Variable-indirected import bypasses Pyright reportMissingImports.
        """

        @contextlib.asynccontextmanager
        async def factory():
            pw_name = "playwright.async_api"
            pw_mod = importlib.import_module(pw_name)
            async_playwright = pw_mod.async_playwright

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=config.headless)
                page = await browser.new_page(
                    viewport={
                        "width": config.viewport_width,
                        "height": config.viewport_height,
                    }
                )
                page.set_default_timeout(config.default_timeout * 1000)
                try:
                    yield page
                finally:
                    await browser.close()

        return cls(config, page_factory=factory)

    async def _ensure_page(self) -> Any:
        if self._page is None:
            if self._page_factory is None:
                raise RuntimeError(
                    "No page_factory provided. Use BrowserClient.from_config() "
                    "for production or pass a mock page_factory for testing."
                )
            self._exit_stack = AsyncExitStack()
            self._page = await self._exit_stack.enter_async_context(
                self._page_factory()
            )
        return self._page

    async def navigate(self, url: str) -> PageSnapshot:
        page = await self._ensure_page()
        await page.goto(url, wait_until="domcontentloaded")
        snapshot = await take_snapshot(page)
        self._ref_map = build_ref_map(snapshot)
        return snapshot

    async def snapshot(self) -> PageSnapshot:
        page = await self._ensure_page()
        snap = await take_snapshot(page)
        self._ref_map = build_ref_map(snap)
        return snap

    async def click(self, ref_or_selector: str) -> dict[str, Any]:
        page = await self._ensure_page()
        locator = self._resolve(page, ref_or_selector)
        url_before = page.url
        await locator.click()

        result: dict[str, Any] = {"clicked": ref_or_selector}
        if page.url != url_before:
            snap = await take_snapshot(page)
            self._ref_map = build_ref_map(snap)
            result["navigated"] = True
            result["snapshot"] = snap.to_text()
        else:
            result["navigated"] = False
            result["snapshot"] = None
        return result

    async def type_text(self, ref_or_selector: str, text: str) -> None:
        page = await self._ensure_page()
        locator = self._resolve(page, ref_or_selector)
        await locator.fill(text)

    async def screenshot(self, path: str | None = None) -> str:
        page = await self._ensure_page()
        if path is None:
            path = tempfile.mktemp(suffix=".png", prefix="browser_")
        await page.screenshot(path=path)
        return path

    async def get_text(self, ref_or_selector: str | None = None) -> str:
        page = await self._ensure_page()
        if ref_or_selector is None:
            return await page.text_content()
        locator = self._resolve(page, ref_or_selector)
        return await locator.text_content()

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._page = None
        self._exit_stack = None
        self._ref_map = {}

    def _resolve(self, page: Any, ref_or_selector: str) -> Any:
        if ref_or_selector.startswith("@e"):
            if ref_or_selector not in self._ref_map:
                raise KeyError(
                    f"Unknown ref {ref_or_selector!r}. "
                    "Call browser_navigate or browser_snapshot to refresh refs."
                )
            role, name = self._ref_map[ref_or_selector]
            return page.get_by_role(role, name=name).first
        return page.locator(ref_or_selector).first
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_browser.py -v`
Expected: 23 passed

- [ ] **Step 6: Commit**

```bash
git add src/topsport_agent/browser/client.py tests/test_browser.py
git commit -m "feat(browser): add BrowserClient with lazy init and injectable factory"
```

---

### Task 4: BrowserToolSource

**Files:**
- Create: `src/topsport_agent/browser/tools.py`
- Modify: `tests/test_browser.py`

- [ ] **Step 1: Write tests for BrowserToolSource**

Append to `tests/test_browser.py`:

```python
from topsport_agent.browser.tools import BrowserToolSource
from topsport_agent.types.tool import ToolContext, ToolSpec


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
        tree = _make_tree([{"role": "button", "name": "Go"}])
        page = MockPage(tree=tree)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        tools = await source.list_tools()
        nav = next(t for t in tools if t.name == "browser_navigate")
        result = await nav.handler({"url": "https://example.com"}, _tool_ctx(cancel_event))
        assert "URL: https://example.com" in result["elements"]
        assert '@e1 [button] "Go"' in result["elements"]

    async def test_click_handler(self, cancel_event: asyncio.Event):
        tree = _make_tree([{"role": "button", "name": "OK"}])
        page = MockPage(tree=tree)
        client = BrowserClient(BrowserConfig(), page_factory=_mock_page_factory(page))
        source = BrowserToolSource(client)

        await client.navigate("https://example.com")
        tools = await source.list_tools()
        click = next(t for t in tools if t.name == "browser_click")
        result = await click.handler({"target": "@e1"}, _tool_ctx(cancel_event))
        assert result["clicked"] == "@e1"

    async def test_type_handler(self, cancel_event: asyncio.Event):
        tree = _make_tree([{"role": "textbox", "name": "Name"}])
        page = MockPage(tree=tree)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_browser.py::TestBrowserToolSource -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topsport_agent.browser.tools'`

- [ ] **Step 3: Implement BrowserToolSource**

```python
# src/topsport_agent/browser/tools.py
"""BrowserToolSource — exposes browser operations as ToolSpecs for the engine."""

from __future__ import annotations

import logging
from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .client import BrowserClient

_logger = logging.getLogger(__name__)


class BrowserToolSource:
    """ToolSource implementation providing 6 browser control tools."""

    name: str = "browser"

    def __init__(self, client: BrowserClient) -> None:
        self._client = client

    async def list_tools(self) -> list[ToolSpec]:
        return [
            self._navigate_spec(),
            self._snapshot_spec(),
            self._click_spec(),
            self._type_spec(),
            self._screenshot_spec(),
            self._get_text_spec(),
        ]

    def _navigate_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                snap = await client.navigate(args["url"])
                return {
                    "url": snap.url,
                    "title": snap.title,
                    "elements": snap.to_text(),
                }
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_navigate",
            description=(
                "Navigate to a URL and return a snapshot of interactive elements. "
                "Each element has a @ref (e.g. @e1) for use with other browser tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to"},
                },
                "required": ["url"],
            },
            handler=handler,
        )

    def _snapshot_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                snap = await client.snapshot()
                return {
                    "url": snap.url,
                    "title": snap.title,
                    "elements": snap.to_text(),
                }
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_snapshot",
            description=(
                "Get a snapshot of interactive elements on the current page. "
                "Each element has a @ref (e.g. @e1) for use with other browser tools."
            ),
            parameters={"type": "object", "properties": {}},
            handler=handler,
        )

    def _click_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                return await client.click(args["target"])
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_click",
            description=(
                "Click an element by @ref (from snapshot) or CSS selector. "
                "If the click triggers navigation, returns the new page snapshot."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "@ref (e.g. '@e3') or CSS selector",
                    },
                },
                "required": ["target"],
            },
            handler=handler,
        )

    def _type_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                await client.type_text(args["target"], args["text"])
                return {"typed": args["text"], "target": args["target"]}
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_type",
            description="Type text into an input element identified by @ref or CSS selector.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "@ref (e.g. '@e1') or CSS selector",
                    },
                    "text": {"type": "string", "description": "Text to type"},
                },
                "required": ["target", "text"],
            },
            handler=handler,
        )

    def _screenshot_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                path = await client.screenshot(args.get("path"))
                return {"path": path}
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_screenshot",
            description="Take a screenshot of the current page. Returns the file path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to save screenshot. Auto-generated if omitted.",
                    },
                },
            },
            handler=handler,
        )

    def _get_text_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                text = await client.get_text(args.get("target"))
                return {"text": text, "target": args.get("target", "page")}
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_get_text",
            description="Get text content from the page or a specific element.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "@ref or CSS selector. Omit for full page text.",
                    },
                },
            },
            handler=handler,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_browser.py -v`
Expected: 32 passed

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/browser/tools.py tests/test_browser.py
git commit -m "feat(browser): add BrowserToolSource with 6 tool definitions"
```

---

### Task 5: Public Exports and Dependency Group

**Files:**
- Modify: `src/topsport_agent/browser/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_browser.py`

- [ ] **Step 1: Write import test**

Add to `tests/test_browser.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_browser.py::TestBrowserPublicAPI -v`
Expected: FAIL — `ImportError: cannot import name 'BrowserClient' from 'topsport_agent.browser'`

- [ ] **Step 3: Update `__init__.py` with public exports**

```python
# src/topsport_agent/browser/__init__.py
"""Built-in browser control module."""

from .client import BrowserClient, PageFactory
from .tools import BrowserToolSource
from .types import BrowserConfig, PageSnapshot, SnapshotEntry

__all__ = [
    "BrowserClient",
    "BrowserConfig",
    "BrowserToolSource",
    "PageFactory",
    "PageSnapshot",
    "SnapshotEntry",
]
```

- [ ] **Step 4: Add `browser` dependency group to `pyproject.toml`**

Add after the existing `llm` group:

```toml
browser = [
    "playwright>=1.40.0",
]
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/test_browser.py -v`
Expected: 33 passed

Then run full suite:

Run: `uv run pytest -v`
Expected: all pass (browser tests need no Playwright since they use mocks)

- [ ] **Step 6: Commit**

```bash
git add src/topsport_agent/browser/__init__.py pyproject.toml tests/test_browser.py
git commit -m "feat(browser): finalize public API and add browser dependency group"
```

---

### Task 6: Full Test Suite Verification and README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: all existing tests + 33 new browser tests pass

- [ ] **Step 2: Update README**

Add a `Browser Control` section to `README.md` describing:
- What the module does
- How to install the optional dependency (`uv sync --group browser && playwright install chromium`)
- Usage example showing BrowserClient + BrowserToolSource integration with Engine

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add browser control module to README"
```
