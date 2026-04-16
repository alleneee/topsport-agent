from __future__ import annotations

from typing import Any

from ..adapters.openai_chat import OpenAIChatAdapter
from ..clients import OpenAIChatClient
from ..provider import LLMResponse
from ..request import LLMRequest


class OpenAIChatProvider:
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
        payload = self._adapter.build_payload(request)
        completion = await self._client.create(payload)
        return self._adapter.parse_response(completion)
