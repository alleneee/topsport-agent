from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types.message import Message
from ..types.tool import ToolSpec


@dataclass(slots=True)
class LLMRequest:
    model: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    max_output_tokens: int | None = None
    temperature: float | None = None
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
