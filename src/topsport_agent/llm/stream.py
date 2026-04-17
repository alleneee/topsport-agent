"""LLM 流式响应类型。

设计要点：
- 流式 chunk 只携带增量（text_delta / tool_call_delta），不重复已发送的内容
- 最终一个 chunk 携带完整 LLMResponse（等价于非流式 complete() 的返回值）
- Engine 可以边接收边 emit LLM_TEXT_DELTA 事件给 UI 层
- 工具调用必须等流结束才完整（参数 JSON 可能分片到达）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .response import LLMResponse

ChunkType = Literal["text_delta", "tool_call_delta", "done"]


@dataclass(slots=True)
class LLMStreamChunk:
    """流式响应的一个增量片段。

    type 三种:
    - "text_delta": 文本增量，text_delta 字段非 None
    - "tool_call_delta": 工具调用元信息到达（名称、id），arguments 分片在同一 chunk 的 tool_delta_args 中
    - "done": 流结束，final_response 字段非 None，包含完整聚合结果
    """

    type: ChunkType
    text_delta: str | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_delta_args: str | None = None
    final_response: LLMResponse | None = None
