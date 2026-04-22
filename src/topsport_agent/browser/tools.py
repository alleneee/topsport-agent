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
            trust_level="untrusted",
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
            trust_level="untrusted",
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
            trust_level="untrusted",
        )
