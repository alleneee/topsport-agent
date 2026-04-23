"""BrowserClient — Playwright wrapper with lazy init and injectable factory."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import tempfile
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any, Literal

from .snapshot import build_ref_map, take_snapshot
from .types import BrowserConfig, PageSnapshot
from .url_policy import BrowserURLPolicy

PageFactory = Callable[[], AbstractAsyncContextManager[Any]]

WaitState = Literal["visible", "hidden", "attached", "detached"]


def _clean_text(raw: str) -> str:
    """Collapse excessive whitespace and blank lines from page text."""
    import re
    # Collapse runs of whitespace (tabs, spaces) into single space
    text = re.sub(r"[^\S\n]+", " ", raw)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class BrowserClient:
    """Session-scoped browser control. Lazy-initialized on first operation.

    Production: use ``from_config()`` to build with real Playwright.
    Tests: pass a mock ``page_factory`` to ``__init__`` directly.

    iframe 支持：``snapshot(frame_selector=...)`` 把后续 click/type/get_text
    的默认作用域切到该 iframe，直到下一次 navigate / click 触发跳转时复位。
    LLM 通常不必每步都重复 frame_selector。
    """

    def __init__(
        self,
        config: BrowserConfig,
        page_factory: PageFactory | None = None,
        *,
        url_policy: BrowserURLPolicy | None = None,
    ) -> None:
        self._config = config
        self._page_factory = page_factory
        # 默认严格策略：http/https + 拒绝 RFC1918 + 拒绝 metadata
        self._url_policy = url_policy or BrowserURLPolicy()
        self._page: Any | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._ref_map: dict[str, tuple[str, str, int | None]] = {}
        self._snapshot_frame: str = ""

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

    def _get_root(self, page: Any, frame_selector: str) -> Any:
        # 显式 frame_selector 优先，否则回退到 snapshot 时记录的作用域。
        effective = frame_selector or self._snapshot_frame
        if not effective:
            return page
        return page.frame_locator(effective)

    async def navigate(self, url: str) -> PageSnapshot:
        # H-S6: 在 playwright goto 之前挡 scheme / 内网 / metadata 目标
        self._url_policy.check(url)
        page = await self._ensure_page()
        # 跨站跳转后 iframe 作用域一律作废，避免 ref 解析漂到旧 frame
        self._snapshot_frame = ""
        await page.goto(url, wait_until="domcontentloaded")
        snapshot = await take_snapshot(page)
        self._ref_map = build_ref_map(snapshot)
        return snapshot

    async def snapshot(self, frame_selector: str = "") -> PageSnapshot:
        page = await self._ensure_page()
        snap = await take_snapshot(page, frame_selector=frame_selector)
        self._ref_map = build_ref_map(snap)
        self._snapshot_frame = frame_selector
        return snap

    async def navigate_back(self) -> PageSnapshot:
        page = await self._ensure_page()
        self._snapshot_frame = ""
        await page.go_back()
        snap = await take_snapshot(page)
        self._ref_map = build_ref_map(snap)
        return snap

    async def click(
        self,
        ref_or_selector: str,
        *,
        frame_selector: str = "",
    ) -> dict[str, Any]:
        page = await self._ensure_page()
        locator = self._resolve(page, ref_or_selector, frame_selector=frame_selector)
        url_before = page.url
        await locator.click()

        result: dict[str, Any] = {"clicked": ref_or_selector}
        if page.url != url_before:
            # 跳转后 frame 作用域失效，重取全页 snapshot
            self._snapshot_frame = ""
            snap = await take_snapshot(page)
            self._ref_map = build_ref_map(snap)
            result["navigated"] = True
            result["snapshot"] = snap.to_text()
        else:
            result["navigated"] = False
            result["snapshot"] = None
        return result

    async def type_text(
        self,
        ref_or_selector: str,
        text: str,
        *,
        frame_selector: str = "",
    ) -> None:
        page = await self._ensure_page()
        locator = self._resolve(page, ref_or_selector, frame_selector=frame_selector)
        await locator.fill(text)

    async def press_key(
        self,
        key: str,
        *,
        target: str | None = None,
        frame_selector: str = "",
    ) -> None:
        """Dispatch a keystroke. Without target → page-level; with target → on that element."""
        page = await self._ensure_page()
        if target is None:
            await page.keyboard.press(key)
            return
        locator = self._resolve(page, target, frame_selector=frame_selector)
        await locator.press(key)

    async def select_option(
        self,
        ref_or_selector: str,
        values: list[str],
        *,
        frame_selector: str = "",
    ) -> list[str]:
        page = await self._ensure_page()
        locator = self._resolve(page, ref_or_selector, frame_selector=frame_selector)
        return await locator.select_option(values)

    async def wait_for(
        self,
        *,
        selector: str | None = None,
        state: WaitState = "visible",
        seconds: float | None = None,
        frame_selector: str = "",
    ) -> None:
        """Wait for a selector state and/or sleep a fixed duration.

        至少提供 selector 或 seconds 其一；两者同时提供时先 sleep 后等状态。
        """
        if selector is None and seconds is None:
            raise ValueError("wait_for requires one of: selector, seconds")
        if seconds is not None and seconds > 0:
            await asyncio.sleep(seconds)
        if selector is not None:
            page = await self._ensure_page()
            root = self._get_root(page, frame_selector)
            await root.locator(selector).first.wait_for(state=state)

    async def screenshot(self, path: str | None = None) -> str:
        page = await self._ensure_page()
        if path is None:
            path = tempfile.mktemp(suffix=".png", prefix="browser_")
        await page.screenshot(path=path)
        return path

    # Selectors tried in order to find the main content area.
    _MAIN_CONTENT_SELECTORS = [
        "main",
        "article",
        "[role='main']",
        "#content",
        ".content",
        ".post-content",
        ".article-content",
        ".entry-content",
    ]

    async def get_text(
        self,
        ref_or_selector: str | None = None,
        *,
        frame_selector: str = "",
    ) -> str:
        page = await self._ensure_page()
        if ref_or_selector is not None:
            locator = self._resolve(page, ref_or_selector, frame_selector=frame_selector)
            return await locator.text_content() or ""

        # 全页取文本时尊重 frame 作用域
        root = self._get_root(page, frame_selector)
        for selector in self._MAIN_CONTENT_SELECTORS:
            loc = root.locator(selector).first
            if await loc.count() > 0:
                text = await loc.text_content() or ""
                if len(text.strip()) > 100:
                    return _clean_text(text)

        return _clean_text(await root.locator("body").text_content() or "")

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._page = None
        self._exit_stack = None
        self._ref_map = {}
        self._snapshot_frame = ""

    def _resolve(
        self,
        page: Any,
        ref_or_selector: str,
        *,
        frame_selector: str = "",
    ) -> Any:
        root = self._get_root(page, frame_selector)
        if ref_or_selector.startswith("@e"):
            if ref_or_selector not in self._ref_map:
                raise KeyError(
                    f"Unknown ref {ref_or_selector!r}. "
                    "Call browser_navigate or browser_snapshot to refresh refs."
                )
            role, name, nth = self._ref_map[ref_or_selector]
            locator = root.get_by_role(role, name=name)
            if nth is not None:
                return locator.nth(nth)
            return locator.first
        return root.locator(ref_or_selector).first
