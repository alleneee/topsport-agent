"""SSE 序列化工具：OpenAI chat chunk 格式 + 通用命名事件格式。"""

from __future__ import annotations

import json
import time
from typing import Any


def make_chat_chunk(
    chat_id: str,
    model: str,
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """生成一个 OpenAI 兼容的 chat.completion.chunk 负载。"""
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }


def sse_data(obj: Any) -> str:
    """把对象编码为单行 SSE data 帧。不写 event 名则默认 'message'。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def sse_event(event_name: str, obj: Any) -> str:
    """带 event 名的 SSE 帧，用于 plan 流等自定义事件。"""
    return f"event: {event_name}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"


SSE_DONE_LINE = "data: [DONE]\n\n"
