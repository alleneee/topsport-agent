from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from ..types.message import ToolCall

# 在 Message.extra 中存取供应商原始响应的键名，用于持久化后的重建和调试。
LLM_RESPONSE_EXTRA_KEY = "llm_response"


#  助手响应的结构化块：保留供应商粒度（text / thinking / tool_use）供 Tracer 和回放使用
class TextResponseBlock(TypedDict, total=False):
    type: Literal["text"]
    text: str


class ThinkingResponseBlock(TypedDict, total=False):
    type: Literal["thinking"]
    thinking: str
    signature: str


class ToolUseResponseBlock(TypedDict, total=False):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]
    raw_arguments: str


AssistantResponseBlock = (
    TextResponseBlock | ThinkingResponseBlock | ToolUseResponseBlock
)


@dataclass(slots=True)
class ProviderResponseMetadata:
    """供应商级元数据：携带完整的 assistant content blocks，支持序列化/反序列化。

    用途：Langfuse 追踪需要 thinking 块；会话持久化后需要从 extra 中还原 tool_call id。
    """

    provider: str
    assistant_blocks: list[AssistantResponseBlock]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "assistant_blocks": [dict(block) for block in self.assistant_blocks],
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> ProviderResponseMetadata | None:
        provider = payload.get("provider")
        assistant_blocks = payload.get("assistant_blocks")
        if not isinstance(provider, str):
            return None
        if not isinstance(assistant_blocks, list):
            return None
        normalized_blocks: list[AssistantResponseBlock] = []
        for block in assistant_blocks:
            if not isinstance(block, Mapping):
                return None
            normalized_blocks.append(dict(block))
        return cls(
            provider=provider,
            assistant_blocks=normalized_blocks,
        )


@dataclass(slots=True)
class LLMResponse:
    """Adapter 解析后的统一响应。

    Engine 只读 text / tool_calls / finish_reason，
    response_metadata 和 raw 留给观测层和调试使用。
    """

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    response_metadata: ProviderResponseMetadata | None = None
    raw: Any = None


def wrap_response_metadata(
    metadata: ProviderResponseMetadata | None,
) -> dict[str, Any]:
    """写入 Message.extra：Engine 在构建助手消息时调用。"""
    if not metadata:
        return {}
    return {LLM_RESPONSE_EXTRA_KEY: metadata.to_dict()}


def get_response_metadata(
    extra: Mapping[str, Any],
) -> ProviderResponseMetadata | None:
    """从 Message.extra 读回：Engine 在重建历史时调用。"""
    metadata = extra.get(LLM_RESPONSE_EXTRA_KEY)
    if not isinstance(metadata, Mapping):
        return None
    return ProviderResponseMetadata.from_dict(metadata)
