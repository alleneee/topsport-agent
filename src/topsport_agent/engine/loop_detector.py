from __future__ import annotations

import hashlib
import json
from typing import Any

from ..types.tool import ToolContext, ToolSpec

LOOP_MESSAGE = "Repeated tool call with identical arguments detected. Try a different approach or different arguments."


class LoopDetector:
    def __init__(self, window: int = 5, threshold: int = 3) -> None:
        self._window = window
        self._threshold = threshold
        self._history: dict[str, list[str]] = {}

    def check(self, session_id: str, tool_name: str, arguments: dict[str, Any]) -> bool:
        sig = self._signature(tool_name, arguments)
        history = self._history.setdefault(session_id, [])
        history.append(sig)
        if len(history) > self._window:
            history.pop(0)
        if len(history) < self._threshold:
            return False
        recent = history[-self._threshold :]
        return len(set(recent)) == 1

    def clear(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    @staticmethod
    def _signature(name: str, args: dict[str, Any]) -> str:
        payload = json.dumps({"n": name, "a": args}, sort_keys=True, default=str)
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def wrap(self, spec: ToolSpec) -> ToolSpec:
        detector = self
        original_handler = spec.handler

        async def guarded_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            if detector.check(ctx.session_id, spec.name, args):
                return {"loop_detected": True, "message": LOOP_MESSAGE}
            return await original_handler(args, ctx)

        return ToolSpec(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            handler=guarded_handler,
        )

    def wrap_all(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        return [self.wrap(spec) for spec in specs]
