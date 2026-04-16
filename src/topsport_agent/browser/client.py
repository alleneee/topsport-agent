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
            return await page.locator("body").text_content() or ""
        locator = self._resolve(page, ref_or_selector)
        return await locator.text_content() or ""

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
