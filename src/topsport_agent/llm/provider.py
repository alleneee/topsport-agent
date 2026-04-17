from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .request import LLMRequest
from .response import LLMResponse
from .stream import LLMStreamChunk


class LLMProvider(Protocol):
    """LLM 供应商的统一契约：Engine 只依赖此 Protocol，不感知底层是 Anthropic 还是 OpenAI。"""

    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


@runtime_checkable
class StreamingLLMProvider(Protocol):
    """可选的流式输出能力。支持流式的 Provider 实现此 Protocol。

    Engine 用 isinstance(provider, StreamingLLMProvider) 检测能力，
    不支持流式的 Provider（如早期 mock）直接忽略此扩展。

    stream 必须以 async generator 形式实现（用 `async def` + `yield`），
    调用方直接 `async for chunk in provider.stream(request)`。
    """

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]: ...
