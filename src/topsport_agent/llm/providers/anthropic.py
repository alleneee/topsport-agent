from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..adapters.anthropic import AnthropicMessagesAdapter
from ..clients import AnthropicMessagesClient
from ..provider import LLMResponse
from ..request import LLMRequest
from ..stream import LLMStreamChunk


class AnthropicProvider:
    """Anthropic 供应商：组合 Adapter（编解码）和 Client（网络调用）。client / adapter 均可注入，测试时无需真实 SDK。"""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        adapter: Any | None = None,
        max_tokens: int = 4096,
        thinking_budget: int | None = None,
    ) -> None:
        self._adapter = adapter or AnthropicMessagesAdapter(
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        self._client = client or AnthropicMessagesClient(
            api_key=api_key,
            base_url=base_url,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """三步流水线：构建 payload -> 发起请求 -> 解析响应。"""
        payload = self._adapter.build_payload(request)
        response = await self._client.create(payload)
        return self._adapter.parse_response(response)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """流式调用：边产出 text_delta，结束时产出 done + 聚合后的 LLMResponse。

        聚合 response 沿用 Adapter.parse_response()，确保流式和非流式输出结构一致。
        """
        payload = self._adapter.build_payload(request)
        final_response: LLMResponse | None = None

        async for event in self._client.stream(payload):
            if event["type"] == "text_delta":
                yield LLMStreamChunk(type="text_delta", text_delta=event["text"])
            elif event["type"] == "final_message":
                final_response = self._adapter.parse_response(event["message"])

        yield LLMStreamChunk(type="done", final_response=final_response)
