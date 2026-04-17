from __future__ import annotations

import logging
from typing import Any

from ...types.session import Session
from .auto import auto_compact
from .micro import micro_compact

_logger = logging.getLogger(__name__)


class CompactionHook:
    name = "compaction"

    def __init__(
        self,
        provider: Any,
        summary_model: str,
        *,
        context_window: int = 100_000,
        threshold: float = 0.65,
        keep_recent_messages: int = 6,
        keep_recent_tool_results: int = 3,
    ) -> None:
        self._provider = provider
        self._summary_model = summary_model
        self._context_window = context_window
        self._threshold = threshold
        self._keep_recent = keep_recent_messages
        self._keep_tool_results = keep_recent_tool_results

    async def after_step(self, session: Session, step: int) -> None:
        """每步结束后两阶段压缩：先 micro（廉价，清工具输出）再 auto（昂贵，LLM 摘要）。"""
        # micro 先跑，缩减 token 数，可能让 auto 阈值判定通过得更晚。
        session.messages[:] = micro_compact(
            session.messages, self._keep_tool_results
        )

        compacted, did_compact = await auto_compact(
            session.messages,
            session_goal=session.goal,
            system_identity=session.system_prompt,
            provider=self._provider,
            summary_model=self._summary_model,
            context_window=self._context_window,
            threshold=self._threshold,
            keep_recent=self._keep_recent,
        )

        if did_compact:
            _logger.info(
                "session %s: auto-compacted at step %d (%d -> %d messages)",
                session.id,
                step,
                len(session.messages),
                len(compacted),
            )
            # 原地替换 session.messages，外部引用同一个列表仍然有效。
            session.messages[:] = compacted
