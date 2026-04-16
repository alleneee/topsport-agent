from __future__ import annotations

from typing import Any

from topsport_agent.llm import LLMRequest, LLMResponse
from topsport_agent.llm.providers import AnthropicProvider, OpenAIChatProvider


class _SpyAdapter:
    def __init__(self, payload: dict[str, Any], response: LLMResponse) -> None:
        self.payload = payload
        self.response = response
        self.seen_requests: list[LLMRequest] = []
        self.seen_raw: list[Any] = []

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        self.seen_requests.append(request)
        return dict(self.payload)

    def parse_response(self, raw_response: Any) -> LLMResponse:
        self.seen_raw.append(raw_response)
        return self.response


class _OpenAISpyClient:
    def __init__(self, raw_response: Any) -> None:
        self.raw_response = raw_response
        self.requests: list[dict[str, Any]] = []

    async def create(self, payload: dict[str, Any]) -> Any:
        self.requests.append(payload)
        return self.raw_response


class _AnthropicSpyClient:
    def __init__(self, raw_response: Any) -> None:
        self.raw_response = raw_response
        self.requests: list[dict[str, Any]] = []

    async def create(self, payload: dict[str, Any]) -> Any:
        self.requests.append(payload)
        return self.raw_response


async def test_openai_provider_uses_adapter_for_payload_and_response():
    request = LLMRequest(model="gpt-5.1", messages=[])
    raw_response = object()
    expected = LLMResponse(text="ok", finish_reason="stop")
    adapter = _SpyAdapter({"model": "gpt-5.1", "messages": []}, expected)
    client = _OpenAISpyClient(raw_response)
    provider = OpenAIChatProvider(client=client, adapter=adapter)

    result = await provider.complete(request)

    assert adapter.seen_requests == [request]
    assert client.requests == [{"model": "gpt-5.1", "messages": []}]
    assert adapter.seen_raw == [raw_response]
    assert result == expected


async def test_anthropic_provider_uses_adapter_for_payload_and_response():
    request = LLMRequest(model="claude-sonnet-4-5", messages=[])
    raw_response = object()
    expected = LLMResponse(text="ok", finish_reason="end_turn")
    adapter = _SpyAdapter(
        {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 128},
        expected,
    )
    client = _AnthropicSpyClient(raw_response)
    provider = AnthropicProvider(client=client, adapter=adapter)

    result = await provider.complete(request)

    assert adapter.seen_requests == [request]
    assert client.requests == [
        {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 128}
    ]
    assert adapter.seen_raw == [raw_response]
    assert result == expected
