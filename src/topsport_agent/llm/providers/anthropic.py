from __future__ import annotations

from typing import Any

from ..adapters.anthropic import AnthropicMessagesAdapter
from ..clients import AnthropicMessagesClient
from ..provider import LLMResponse
from ..request import LLMRequest


class AnthropicProvider:
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
        payload = self._adapter.build_payload(request)
        response = await self._client.create(payload)
        return self._adapter.parse_response(response)
