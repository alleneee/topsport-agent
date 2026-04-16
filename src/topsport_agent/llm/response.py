from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from ..types.message import ToolCall

LLM_RESPONSE_EXTRA_KEY = "llm_response"


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
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    response_metadata: ProviderResponseMetadata | None = None
    raw: Any = None


def wrap_response_metadata(
    metadata: ProviderResponseMetadata | None,
) -> dict[str, Any]:
    if not metadata:
        return {}
    return {LLM_RESPONSE_EXTRA_KEY: metadata.to_dict()}


def get_response_metadata(
    extra: Mapping[str, Any],
) -> ProviderResponseMetadata | None:
    metadata = extra.get(LLM_RESPONSE_EXTRA_KEY)
    if not isinstance(metadata, Mapping):
        return None
    return ProviderResponseMetadata.from_dict(metadata)
