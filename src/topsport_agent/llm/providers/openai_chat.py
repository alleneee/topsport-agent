from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..adapters.openai_chat import OpenAIChatAdapter
from ..clients import OpenAIChatClient
from ..provider import LLMResponse
from ..request import LLMRequest
from ..stream import LLMStreamChunk


class OpenAIChatProvider:
    """OpenAI Chat 供应商：与 AnthropicProvider 结构对称，组合 Adapter + Client。

    额外支持 organization 和 reasoning_effort（o 系列模型）。
    """

    name = "openai-chat"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        client: Any | None = None,
        adapter: Any | None = None,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> None:
        self._adapter = adapter or OpenAIChatAdapter(
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        self._client = client or OpenAIChatClient(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """三步流水线：构建 payload -> 发起请求 -> 解析响应。"""
        payload = self._adapter.build_payload(request)
        completion = await self._client.create(payload)
        return self._adapter.parse_response(completion)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """流式调用：边产出 text_delta，结束时产出 done + 聚合后的 LLMResponse。"""
        payload = self._adapter.build_payload(request)
        final_response: LLMResponse | None = None

        async for event in self._client.stream(payload):
            if event["type"] == "text_delta":
                yield LLMStreamChunk(type="text_delta", text_delta=event["text"])
            elif event["type"] == "final_completion":
                final_response = self._adapter.parse_response(event["completion"])

        yield LLMStreamChunk(type="done", final_response=final_response)
