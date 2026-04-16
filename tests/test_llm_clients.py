from __future__ import annotations

from types import SimpleNamespace

from topsport_agent.llm.clients import AnthropicMessagesClient, OpenAIChatClient


class _OpenAIRecorder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **payload):
        self.requests.append(payload)
        return {"ok": True}


class _AnthropicRecorder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **payload):
        self.requests.append(payload)
        return {"ok": True}


class _RetryableError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status={status_code}")
        self.status_code = status_code


class _FlakyOpenAIRecorder:
    failures: list[Exception] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests: list[dict] = []
        self._failures = list(type(self).failures)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **payload):
        self.requests.append(payload)
        if self._failures:
            raise self._failures.pop(0)
        return {"ok": True}


class _FlakyAnthropicRecorder:
    failures: list[Exception] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests: list[dict] = []
        self._failures = list(type(self).failures)
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **payload):
        self.requests.append(payload)
        if self._failures:
            raise self._failures.pop(0)
        return {"ok": True}


def test_openai_client_prefers_explicit_config_over_env(monkeypatch):
    module = SimpleNamespace(AsyncOpenAI=_OpenAIRecorder)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "env-org")

    client = OpenAIChatClient(
        api_key="explicit-key",
        base_url="https://explicit.example.com",
        organization="explicit-org",
        module=module,
    )

    assert client._client.kwargs == {
        "api_key": "explicit-key",
        "base_url": "https://explicit.example.com",
        "organization": "explicit-org",
    }


async def test_openai_client_forwards_payload_to_sdk():
    module = SimpleNamespace(AsyncOpenAI=_OpenAIRecorder)
    client = OpenAIChatClient(module=module)

    result = await client.create({"model": "gpt-5.1", "messages": []})

    assert client._client.requests == [{"model": "gpt-5.1", "messages": []}]
    assert result == {"ok": True}


def test_anthropic_client_reads_env_defaults(monkeypatch):
    module = SimpleNamespace(AsyncAnthropic=_AnthropicRecorder)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.example.com")

    client = AnthropicMessagesClient(module=module)

    assert client._client.kwargs == {
        "api_key": "env-key",
        "base_url": "https://env.example.com",
    }


async def test_anthropic_client_forwards_payload_to_sdk():
    module = SimpleNamespace(AsyncAnthropic=_AnthropicRecorder)
    client = AnthropicMessagesClient(module=module)

    result = await client.create({"model": "claude-sonnet-4-5", "messages": []})

    assert client._client.requests == [
        {"model": "claude-sonnet-4-5", "messages": []}
    ]
    assert result == {"ok": True}


async def test_openai_client_retries_retryable_status_errors():
    _FlakyOpenAIRecorder.failures = [_RetryableError(529), _RetryableError(429)]
    module = SimpleNamespace(AsyncOpenAI=_FlakyOpenAIRecorder)
    client = OpenAIChatClient(
        module=module,
        retry_base_delay=0,
    )

    result = await client.create({"model": "gpt-5.1", "messages": []})

    assert client._client.requests == [
        {"model": "gpt-5.1", "messages": []},
        {"model": "gpt-5.1", "messages": []},
        {"model": "gpt-5.1", "messages": []},
    ]
    assert result == {"ok": True}


async def test_anthropic_client_retries_retryable_status_errors():
    _FlakyAnthropicRecorder.failures = [_RetryableError(529)]
    module = SimpleNamespace(AsyncAnthropic=_FlakyAnthropicRecorder)
    client = AnthropicMessagesClient(
        module=module,
        retry_base_delay=0,
    )

    result = await client.create({"model": "claude-sonnet-4-5", "messages": []})

    assert client._client.requests == [
        {"model": "claude-sonnet-4-5", "messages": []},
        {"model": "claude-sonnet-4-5", "messages": []},
    ]
    assert result == {"ok": True}
