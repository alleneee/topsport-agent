# Browser Control Module Design

Built-in browser control for topsport-agent, using Playwright as the underlying driver.

## Goals

- Agent can navigate web pages, interact with elements, take screenshots, and extract text
- Snapshot + ref interaction model: LLM uses `@e1`, `@e2` refs instead of CSS selectors
- Session-level singleton browser: state persists across engine steps
- Playwright as optional dependency group, tests run without it
- Follow existing project patterns (MCP module structure, injectable factory)

## Non-Goals

- No MCP server exposure (internal module only)
- No advanced tools (network intercept, PDF export, cookies) in this iteration
- No multi-tab / multi-page management
- No headed mode or user handoff

## Module Structure

```
src/topsport_agent/browser/
  __init__.py          # Public API: BrowserClient, BrowserToolSource, BrowserConfig
  client.py            # BrowserClient — Playwright lifecycle, page operations
  snapshot.py           # Accessibility tree parsing, ref assignment, snapshot diffing
  tools.py              # BrowserToolSource(ToolSource) — 6 ToolSpec definitions
  types.py              # BrowserConfig, ElementRef, PageSnapshot, SnapshotEntry
```

## Types (`types.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class BrowserConfig:
    """Browser launch configuration."""
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    default_timeout: float = 30.0   # seconds, for navigation and waits


@dataclass(slots=True)
class SnapshotEntry:
    """A single interactive element in the page snapshot."""
    ref: str                         # e.g. "@e1"
    role: str                        # ARIA role: "button", "textbox", "link", etc.
    name: str                        # Accessible name / label
    tag: str = ""                    # HTML tag name
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PageSnapshot:
    """Structured snapshot of interactive elements on the current page."""
    url: str
    title: str
    entries: list[SnapshotEntry] = field(default_factory=list)

    def to_text(self) -> str:
        """Format as LLM-readable text block."""
        lines = [f"URL: {self.url}", f"Title: {self.title}", ""]
        for e in self.entries:
            name_part = f' "{e.name}"' if e.name else ""
            lines.append(f"  {e.ref} [{e.role}]{name_part}")
        return "\n".join(lines)
```

## BrowserClient (`client.py`)

### Design Principles

1. **Injectable factory** — constructor takes a `page_factory` callable, same pattern as
   `MCPClient(session_factory=...)`. Tests inject mock page; production uses `from_config`.
2. **Lazy initialization** — browser/page created on first operation, not at construction.
3. **Session-scoped lifecycle** — one `BrowserClient` per session; call `close()` at session end.
4. **All public methods are async** — natural fit with Playwright's async API.

### Interface

```python
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager

PageFactory = Callable[[], AbstractAsyncContextManager[Any]]

class BrowserClient:
    def __init__(self, config: BrowserConfig, page_factory: PageFactory | None = None):
        ...

    @classmethod
    def from_config(cls, config: BrowserConfig) -> BrowserClient:
        """Production entry: creates real Playwright page_factory."""
        ...

    # --- Page Operations ---

    async def navigate(self, url: str) -> PageSnapshot:
        """Navigate to URL, wait for load, return snapshot."""

    async def snapshot(self) -> PageSnapshot:
        """Snapshot current page interactive elements with @refs."""

    async def click(self, ref_or_selector: str) -> dict[str, Any]:
        """Click element by @ref or CSS selector.
        Returns: {"navigated": bool, "snapshot": PageSnapshot | None}
        If click triggers navigation, auto-attaches new snapshot.
        """

    async def type_text(self, ref_or_selector: str, text: str) -> None:
        """Clear and type text into input element."""

    async def screenshot(self, path: str | None = None) -> str:
        """Take screenshot. Returns file path (auto-generated if not provided)."""

    async def get_text(self, ref_or_selector: str | None = None) -> str:
        """Get text content. None = full page, otherwise scoped to element."""

    async def close(self) -> None:
        """Close browser and release resources."""

    # --- Internal ---

    def _resolve_ref(self, ref_or_selector: str) -> str:
        """Convert @ref to CSS selector using last snapshot's ref map."""
```

### Ref Resolution

The client maintains a `dict[str, str]` mapping `@ref -> playwright locator strategy`.
Each `snapshot()` call rebuilds this map. Resolution strategy:

1. If input starts with `@e` — look up in ref map, raise if not found
2. Otherwise — treat as CSS selector (passthrough)

### Playwright Integration (in `from_config`)

```python
@classmethod
def from_config(cls, config: BrowserConfig) -> BrowserClient:
    """Lazy Playwright import via variable indirection (Pyright bypass)."""

    @contextlib.asynccontextmanager
    async def factory():
        pw_name = "playwright.async_api"
        pw_mod = importlib.import_module(pw_name)
        async_playwright = pw_mod.async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=config.headless)
            page = await browser.new_page(
                viewport={"width": config.viewport_width,
                           "height": config.viewport_height}
            )
            page.set_default_timeout(config.default_timeout * 1000)
            try:
                yield page
            finally:
                await browser.close()

    return cls(config, page_factory=factory)
```

### Lazy Initialization

The client does NOT open a browser at construction. The first call to any page operation
triggers `page_factory()` via an internal `_ensure_page()` method. This avoids paying
startup cost until the agent actually uses browser tools.

```python
async def _ensure_page(self) -> Any:
    if self._page is None:
        self._exit_stack = AsyncExitStack()
        self._page = await self._exit_stack.enter_async_context(
            self._page_factory()
        )
    return self._page
```

## Snapshot Engine (`snapshot.py`)

### Accessibility Tree Extraction

Playwright exposes `page.accessibility.snapshot()` which returns the full accessibility
tree as a dict. We walk this tree to extract interactive elements.

```python
INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "radio",
    "combobox", "menuitem", "tab", "switch", "searchbox",
    "slider", "spinbutton", "option", "menuitemcheckbox",
    "menuitemradio",
})

async def take_snapshot(page: Any) -> PageSnapshot:
    """Extract interactive elements from accessibility tree, assign @refs."""
    tree = await page.accessibility.snapshot()
    url = page.url
    title = await page.title()

    entries: list[SnapshotEntry] = []
    counter = 0

    def walk(node: dict) -> None:
        nonlocal counter
        role = node.get("role", "")
        if role in INTERACTIVE_ROLES:
            counter += 1
            entries.append(SnapshotEntry(
                ref=f"@e{counter}",
                role=role,
                name=node.get("name", ""),
            ))
        for child in node.get("children", []):
            walk(child)

    if tree:
        walk(tree)

    return PageSnapshot(url=url, title=title, entries=entries)
```

### Ref-to-Locator Mapping

After building the snapshot, we also need a way to locate the actual DOM element for
each ref. Strategy: use `page.get_by_role(role, name=name)` as the primary locator.

```python
def build_ref_map(snapshot: PageSnapshot) -> dict[str, tuple[str, str]]:
    """Returns {ref: (role, name)} for locator resolution."""
    return {e.ref: (e.role, e.name) for e in snapshot.entries}
```

When resolving a ref for interaction:
```python
locator = page.get_by_role(role, name=name)
# If multiple matches, use .first to pick the first one
```

If `get_by_role` returns zero matches (DOM changed since snapshot), raise a clear error
telling the agent to re-snapshot.

## BrowserToolSource (`tools.py`)

Implements `ToolSource` protocol. Returns 6 `ToolSpec` instances.

### Tool Definitions

#### 1. `browser_navigate`

```python
ToolSpec(
    name="browser_navigate",
    description="Navigate to a URL and return a snapshot of interactive elements.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to navigate to"}
        },
        "required": ["url"],
    },
    handler=...,
)
```

Returns: `{"url": str, "title": str, "elements": str}` where elements is the
formatted snapshot text.

#### 2. `browser_snapshot`

```python
ToolSpec(
    name="browser_snapshot",
    description="Get a snapshot of interactive elements on the current page. "
                "Each element has a @ref (e.g. @e1) for use with other browser tools.",
    parameters={"type": "object", "properties": {}},
    handler=...,
)
```

Returns: `{"url": str, "title": str, "elements": str}`

#### 3. `browser_click`

```python
ToolSpec(
    name="browser_click",
    description="Click an element by @ref (from snapshot) or CSS selector. "
                "If the click triggers navigation, returns the new page snapshot.",
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "@ref (e.g. '@e3') or CSS selector",
            }
        },
        "required": ["target"],
    },
    handler=...,
)
```

Returns: `{"clicked": str, "navigated": bool, "snapshot": str | None}`

#### 4. `browser_type`

```python
ToolSpec(
    name="browser_type",
    description="Type text into an input element identified by @ref or CSS selector.",
    parameters={
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "@ref or CSS selector"},
            "text": {"type": "string", "description": "Text to type"},
        },
        "required": ["target", "text"],
    },
    handler=...,
)
```

Returns: `{"typed": str, "target": str}`

#### 5. `browser_screenshot`

```python
ToolSpec(
    name="browser_screenshot",
    description="Take a screenshot of the current page. Returns the file path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to save screenshot. Auto-generated if omitted.",
            }
        },
    },
    handler=...,
)
```

Returns: `{"path": str}`

#### 6. `browser_get_text`

```python
ToolSpec(
    name="browser_get_text",
    description="Get text content from the page or a specific element.",
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "@ref or CSS selector. Omit for full page text.",
            }
        },
    },
    handler=...,
)
```

Returns: `{"text": str, "target": str}`

### ToolSource Implementation

```python
class BrowserToolSource:
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
```

Each `_*_spec()` method returns a `ToolSpec` whose handler delegates to
`self._client.<method>()`, wrapped in try/except to return error dicts
(same pattern as MCPToolSource).

## Lifecycle Management

### Integration with Engine

```python
# Usage in application code:
config = BrowserConfig(headless=True)
browser_client = BrowserClient.from_config(config)
browser_tools = BrowserToolSource(browser_client)

engine = Engine(
    provider=llm_provider,
    tools=[...builtin_tools...],
    tool_sources=[browser_tools],  # Dynamic tool source
)

# After engine run completes:
await browser_client.close()
```

### Cleanup via EventSubscriber (Optional)

For automatic cleanup, register an EventSubscriber that closes the browser on RUN_END:

```python
class BrowserCleanup(EventSubscriber):
    name = "browser_cleanup"

    def __init__(self, client: BrowserClient):
        self._client = client

    async def on_event(self, event: Event) -> None:
        if event.type == EventType.RUN_END:
            await self._client.close()
```

## Dependency Management

### pyproject.toml addition

```toml
[dependency-groups]
browser = [
    "playwright>=1.40.0",
]
```

After `uv sync --group browser`, user must also run `playwright install chromium`
to download the browser binary.

### Import Strategy

Same pattern as MCP and Langfuse: variable-indirected `importlib.import_module()`.

```python
pw_name = "playwright.async_api"
pw_mod = importlib.import_module(pw_name)
```

Only triggered in `BrowserClient.from_config()`. Direct construction via
`BrowserClient(config, page_factory=mock)` never imports playwright.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Playwright not installed | `from_config` raises `ModuleNotFoundError` with clear message |
| Browser binary not installed | Playwright raises its own error on launch |
| Navigation timeout | Handler catches, returns `{"is_error": True, "error": "..."}` |
| Stale ref (DOM changed) | Handler returns error suggesting re-snapshot |
| Element not interactable | Handler catches Playwright error, returns descriptive error |
| Browser already closed | `_ensure_page` re-creates if needed, or returns error |

All errors follow the project pattern: handler catches exceptions and returns
error dicts; engine ToolResult.is_error is reserved for engine-level failures.

## Testing Strategy

### Unit Tests (no Playwright required)

All tests use mock page objects injected via `page_factory`:

```python
@contextlib.asynccontextmanager
async def mock_page_factory():
    yield MockPage(...)

client = BrowserClient(BrowserConfig(), page_factory=mock_page_factory)
```

### Test Cases

1. **client.py**
   - `test_lazy_init_no_browser_until_first_call`
   - `test_navigate_returns_snapshot`
   - `test_click_with_ref_resolves_correctly`
   - `test_click_triggers_navigation_returns_new_snapshot`
   - `test_stale_ref_raises_error`
   - `test_close_releases_resources`
   - `test_css_selector_passthrough`

2. **snapshot.py**
   - `test_snapshot_extracts_interactive_elements`
   - `test_snapshot_ignores_non_interactive_roles`
   - `test_ref_numbering_sequential`
   - `test_empty_page_returns_empty_snapshot`
   - `test_snapshot_to_text_format`

3. **tools.py**
   - `test_tool_source_returns_six_tools`
   - `test_navigate_handler_returns_elements`
   - `test_click_handler_error_returns_dict`
   - `test_screenshot_handler_returns_path`
   - `test_tool_names_no_collision_with_builtins`

4. **Integration** (requires `--group browser`, marked with `pytest.mark.skipif`)
   - `test_real_navigate_and_snapshot` against a local HTML fixture

## Example Agent Interaction

```
User: "Go to https://news.ycombinator.com and find the top story title"

Agent calls: browser_navigate(url="https://news.ycombinator.com")
-> Returns:
   URL: https://news.ycombinator.com
   Title: Hacker News
   
     @e1 [link] "Hacker News"
     @e2 [link] "new"
     @e3 [link] "past"
     @e4 [link] "comments"
     @e5 [link] "Show HN: My Cool Project"
     @e6 [link] "142 comments"
     ...

Agent calls: browser_get_text(target="@e5")
-> Returns: {"text": "Show HN: My Cool Project", "target": "@e5"}

Agent responds: "The top story is 'Show HN: My Cool Project'."
```

## Future Extensions (out of scope)

- `browser_select` — dropdown selection
- `browser_scroll` — page scrolling
- `browser_wait` — wait for element/condition
- `browser_evaluate` — execute arbitrary JavaScript
- `BrowserContextProvider` — auto-inject page state per step
- Snapshot diffing (`snapshot -D` equivalent)
- Annotated screenshots with element labels
