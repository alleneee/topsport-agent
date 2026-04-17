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


# ---------------------------------------------------------------------------
# H-R1 · 流式路径重试对齐：首 delta 前可重试；已 yield 后拒绝重试
# ---------------------------------------------------------------------------


class _AnthropicStreamCtx:
    """模拟 Anthropic SDK 的 stream() async context manager。"""

    def __init__(self, chunks: list[str], raise_in_stream: Exception | None = None,
                 raise_before_start: Exception | None = None) -> None:
        self._chunks = chunks
        self._raise_in_stream = raise_in_stream
        self._raise_before_start = raise_before_start

    async def __aenter__(self):
        if self._raise_before_start is not None:
            raise self._raise_before_start
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self):
        chunks = self._chunks
        err = self._raise_in_stream

        async def _gen():
            for i, c in enumerate(chunks):
                yield c
                if err is not None and i == 0:
                    raise err

        return _gen()

    async def get_final_message(self):
        return SimpleNamespace(content=[], stop_reason="end_turn")


class _FlakyAnthropicStream:
    """messages.stream() 返回的 ctx 每次可配置不同失败行为。"""

    def __init__(self, sequence: list) -> None:
        # sequence 元素可以是异常或 (chunks, raise_in_stream) 元组
        self.sequence = list(sequence)
        self.call_count = 0

    def stream(self, **payload):
        self.call_count += 1
        item = self.sequence.pop(0)
        if isinstance(item, Exception):
            return _AnthropicStreamCtx(chunks=[], raise_before_start=item)
        chunks, raise_in_stream = item
        return _AnthropicStreamCtx(chunks=chunks, raise_in_stream=raise_in_stream)


async def test_anthropic_stream_retries_before_first_delta() -> None:
    """连接建立阶段的瞬态错误 → 重试后客户端看到完整 stream。"""
    sdk_client = SimpleNamespace(
        messages=_FlakyAnthropicStream(
            sequence=[
                _RetryableError(529),          # 第一次 __aenter__ 抛
                (["hello", " world"], None),   # 第二次成功
            ]
        )
    )
    client = AnthropicMessagesClient(sdk_client=sdk_client, retry_base_delay=0)

    events = [e async for e in client.stream({"model": "claude", "messages": []})]
    deltas = [e["text"] for e in events if e["type"] == "text_delta"]
    assert deltas == ["hello", " world"]
    assert sdk_client.messages.call_count == 2


async def test_anthropic_stream_does_not_retry_after_yield() -> None:
    """首个 delta yield 后的错误原样抛出，不再重试。"""
    sdk_client = SimpleNamespace(
        messages=_FlakyAnthropicStream(
            sequence=[
                # 先 yield 一个 chunk 再抛 retryable，按规则仍不重试
                (["partial"], _RetryableError(529)),
            ]
        )
    )
    client = AnthropicMessagesClient(sdk_client=sdk_client, retry_base_delay=0)

    collected: list[str] = []
    gen = client.stream({"model": "claude", "messages": []})
    import pytest as _pytest
    with _pytest.raises(_RetryableError):
        async for e in gen:
            if e["type"] == "text_delta":
                collected.append(e["text"])
    assert collected == ["partial"]
    assert sdk_client.messages.call_count == 1


async def test_anthropic_stream_non_retryable_propagates() -> None:
    """非瞬态错误立即抛，不重试。"""
    sdk_client = SimpleNamespace(
        messages=_FlakyAnthropicStream(
            sequence=[ValueError("permanent")],
        )
    )
    client = AnthropicMessagesClient(sdk_client=sdk_client, retry_base_delay=0)

    import pytest as _pytest
    with _pytest.raises(ValueError, match="permanent"):
        async for _ in client.stream({"model": "claude", "messages": []}):
            pass
    assert sdk_client.messages.call_count == 1


class _FlakyOpenAIStreamCreate:
    """OpenAI stream 接口：create(stream=True) 返回 async iterator 或抛异常。"""

    def __init__(self, sequence: list) -> None:
        # sequence 元素：异常 或 chunks list
        self.sequence = list(sequence)
        self.call_count = 0

    async def create(self, **payload):
        self.call_count += 1
        item = self.sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        # 返回一个 async iterator：按 chunks 发送 delta
        async def _iter():
            for c in item:
                yield SimpleNamespace(
                    model="gpt-5.1",
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content=c, tool_calls=None),
                        finish_reason=None,
                    )],
                    usage=None,
                )
            # 最终 chunk：带 finish_reason + usage
            yield SimpleNamespace(
                model="gpt-5.1",
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            )

        return _iter()


async def test_openai_stream_retries_before_first_delta() -> None:
    sdk_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FlakyOpenAIStreamCreate(
            sequence=[_RetryableError(503), ["hi", " there"]]
        ))
    )
    client = OpenAIChatClient(sdk_client=sdk_client, retry_base_delay=0)

    events = [e async for e in client.stream({"model": "gpt-5.1", "messages": []})]
    deltas = [e["text"] for e in events if e["type"] == "text_delta"]
    assert deltas == ["hi", " there"]
    assert sdk_client.chat.completions.call_count == 2


async def test_openai_stream_does_not_retry_after_yield() -> None:
    class _PartialThenFail:
        def __init__(self):
            self.call_count = 0

        async def create(self, **payload):
            self.call_count += 1

            async def _iter():
                yield SimpleNamespace(
                    model="gpt-5.1",
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content="partial", tool_calls=None),
                        finish_reason=None,
                    )],
                    usage=None,
                )
                raise _RetryableError(529)

            return _iter()

    completions = _PartialThenFail()
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = OpenAIChatClient(sdk_client=sdk_client, retry_base_delay=0)

    collected: list[str] = []
    import pytest as _pytest
    with _pytest.raises(_RetryableError):
        async for e in client.stream({"model": "gpt-5.1", "messages": []}):
            if e["type"] == "text_delta":
                collected.append(e["text"])
    assert collected == ["partial"]
    assert completions.call_count == 1
