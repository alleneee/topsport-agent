from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    """ToolCall 是 LLM 发出的工具调用指令，arguments 已解析为 dict（OpenAI 适配器负责 JSON 反序列化）。"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    """ToolResult 是工具执行后的应答。is_error 仅标记引擎层面的失败，MCP 语义错误通过 output 内容区分。"""
    call_id: str
    output: Any
    is_error: bool = False


@dataclass(slots=True)
class Message:
    """Message 是会话的基本单元。

    核心不变量：带 tool_calls 的 assistant 消息后必须紧跟 tool_results，
    中间不允许插入其他消息类型，否则 Anthropic/OpenAI 均返回 400。
    """
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
