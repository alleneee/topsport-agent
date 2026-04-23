"""BrowserToolSource — exposes browser operations as ToolSpecs for the engine."""

from __future__ import annotations

import logging
from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .client import BrowserClient

_logger = logging.getLogger(__name__)


class BrowserToolSource:
    """ToolSource implementation providing browser control tools.

    Toolset: navigate / back / snapshot / click / type / press_key / select_option /
    wait_for / screenshot / get_text. ``browser_snapshot`` 支持 ``frame_selector``
    切到 iframe 作用域，后续的 click / type / get_text / press_key / select_option
    都会继承这个作用域，直到下一次 navigate 或 click 触发跳转。
    """

    name: str = "browser"

    def __init__(self, client: BrowserClient) -> None:
        self._client = client

    async def list_tools(self) -> list[ToolSpec]:
        return [
            self._navigate_spec(),
            self._back_spec(),
            self._snapshot_spec(),
            self._click_spec(),
            self._type_spec(),
            self._press_key_spec(),
            self._select_option_spec(),
            self._wait_for_spec(),
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

    def _back_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                snap = await client.navigate_back()
                return {
                    "url": snap.url,
                    "title": snap.title,
                    "elements": snap.to_text(),
                }
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_back",
            description="Navigate back to the previous page in history and return a fresh snapshot.",
            parameters={"type": "object", "properties": {}},
            handler=handler,
            trust_level="untrusted",
        )

    def _snapshot_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                frame = args.get("frame_selector", "") or ""
                snap = await client.snapshot(frame_selector=frame)
                out: dict[str, Any] = {
                    "url": snap.url,
                    "title": snap.title,
                    "elements": snap.to_text(),
                }
                if snap.frame_selector:
                    out["frame_selector"] = snap.frame_selector
                return out
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_snapshot",
            description=(
                "Get a snapshot of interactive elements on the current page. "
                "Each element has a @ref (e.g. @e1) for use with other browser tools. "
                "Pass frame_selector (e.g. 'iframe#main') to scope into an iframe; "
                "subsequent click/type/get_text inherit that frame until the next navigation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "frame_selector": {
                        "type": "string",
                        "description": "Optional CSS selector of an iframe to operate inside.",
                    },
                },
            },
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

    def _press_key_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                target = args.get("target")
                await client.press_key(args["key"], target=target)
                return {"pressed": args["key"], "target": target or "page"}
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_press_key",
            description=(
                "Press a keyboard key or chord (e.g. 'Enter', 'Escape', 'Control+A'). "
                "Without target, dispatched at the page level; with a @ref or CSS "
                "selector, the element is focused first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key or chord, e.g. 'Enter', 'Control+A'.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional @ref or CSS selector to focus first.",
                    },
                },
                "required": ["key"],
            },
            handler=handler,
        )

    def _select_option_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                raw = args["values"]
                values = [raw] if isinstance(raw, str) else list(raw)
                selected = await client.select_option(args["target"], values)
                return {"target": args["target"], "selected": selected}
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_select_option",
            description="Select one or more options in a <select> element.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "@ref or CSS selector of the <select>.",
                    },
                    "values": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "Option value or list of values to select.",
                    },
                },
                "required": ["target", "values"],
            },
            handler=handler,
        )

    def _wait_for_spec(self) -> ToolSpec:
        client = self._client

        async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
            try:
                await client.wait_for(
                    selector=args.get("selector"),
                    state=args.get("state", "visible"),
                    seconds=args.get("seconds"),
                )
                return {"waited": True}
            except Exception as exc:
                return {"is_error": True, "error": f"{type(exc).__name__}: {exc}"}

        return ToolSpec(
            name="browser_wait_for",
            description=(
                "Wait for a selector to reach a state, or sleep a number of seconds. "
                "Provide at least one of: selector, seconds."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to wait for.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["visible", "hidden", "attached", "detached"],
                        "description": "Target state for the selector (default 'visible').",
                    },
                    "seconds": {
                        "type": "number",
                        "description": "Additional sleep duration in seconds.",
                    },
                },
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
