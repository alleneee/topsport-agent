from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from ..types.tool import ToolContext, ToolSpec

LOOP_MESSAGE = "Repeated tool call with identical arguments detected. Try a different approach or different arguments."


class LoopDetector:
    """滑动窗口内检测连续相同签名的工具调用，防止 LLM 陷入死循环。"""

    def __init__(
        self, window: int = 5, threshold: int = 3, max_sessions: int = 1000
    ) -> None:
        self._window = window
        self._threshold = threshold
        self._max_sessions = max_sessions
        self._history: OrderedDict[str, list[str]] = OrderedDict()

    def check(self, session_id: str, tool_name: str, arguments: dict[str, Any]) -> bool:
        sig = self._signature(tool_name, arguments)
        if session_id in self._history:
            self._history.move_to_end(session_id)
        history = self._history.setdefault(session_id, [])
        # LRU 淘汰最久未活跃的 session，防止长期运行时内存无限增长。
        if len(self._history) > self._max_sessions:
            self._history.popitem(last=False)
        history.append(sig)
        if len(history) > self._window:
            history.pop(0)
        if len(history) < self._threshold:
            return False
        # 最近 N 次调用签名完全相同即判定为循环。
        recent = history[-self._threshold :]
        return len(set(recent)) == 1

    def clear(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    @staticmethod
    def _signature(name: str, args: dict[str, Any]) -> str:
        payload = json.dumps({"n": name, "a": args}, sort_keys=True, default=str)
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def wrap(self, spec: ToolSpec) -> ToolSpec:
        """用装饰器模式包装 handler：检测到循环时直接返回提示，不执行真正的工具逻辑。"""
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
