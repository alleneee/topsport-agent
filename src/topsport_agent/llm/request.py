from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types.message import Message
from ..types.tool import ToolSpec


@dataclass(slots=True)
class LLMRequest:
    """供应商无关的 LLM 调用描述，Engine 侧构建、Adapter 侧消费。

    provider_options 按供应商 key 隔离（如 {"anthropic": {...}, "openai": {...}}），
    Adapter 只取自己那份，互不干扰。
    """

    model: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    max_output_tokens: int | None = None
    temperature: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
