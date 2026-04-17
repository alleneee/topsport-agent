from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class StructuredOutputProvider(Protocol):
    """可选的结构化输出能力。H-A4：Planner 等需要 JSON schema 结果的调用方不再
    绑死在 tool-call ABI 上——Gemini/Bedrock/原生 JSON mode 都可以实现此 Protocol，
    Planner 通过 isinstance 检测并优先走该路径，兜底仍是 tool-call emulation。

    complete_structured 必须返回符合 schema 的 dict（可以是嵌套结构）。
    实现侧决定如何让 LLM 产生这个结构（tool-call / json_mode / function_call 等）。
    """

    name: str

    async def complete_structured(
        self,
        request: LLMRequest,
        schema: dict[str, Any],
        *,
        tool_name: str = "structured_output",
    ) -> dict[str, Any]: ...
