"""MCP `sampling` capability tests.

覆盖：
- SamplingMessage/Request/Result frozen dataclass
- LLMProviderSamplingHandler 默认 model + 选 hint + allowlist 过滤
- LLMProviderSamplingHandler max_tokens_cap 截顶
- LLMProviderSamplingHandler system_prompt 转 Message Role.SYSTEM
- LLMProviderSamplingHandler 校验 default_model 非空 / max_tokens_cap > 0
- from_sdk_params 转换 + 非 text 内容 raise
- to_sdk_result + to_sdk_error
- MCPClient.set_sampling_handler 持久 + sampling_callback adapter
- adapter 在无 handler 时回 ErrorData -32601
- adapter 在 from_sdk_params 失败时回 ErrorData -32602
- adapter 在 handler 抛异常时回 ErrorData -32603 + log warning
- session_factory attach sampling_callback only when handler 设
- ServerConfig env wiring
- lifespan 集成：enable_mcp_sampling=true 但 model 缺 → fail-fast
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from starlette.testclient import TestClient

from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.mcp import (
    LLMProviderSamplingHandler,
    MCPClient,
    MCPManager,
    MCPServerConfig,
    SamplingMessage,
    SamplingRequest,
    SamplingResult,
)
from topsport_agent.mcp.sampling import (
    RateLimitExceeded,
    TokenBucketRateLimit,
    from_sdk_params,
    to_sdk_error,
    to_sdk_result,
)
from topsport_agent.mcp.types import MCPTransport
from topsport_agent.server.app import _build_mcp_manager, create_app
from topsport_agent.server.config import ServerConfig


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubProvider:
    name = "stub"

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []
        self.next_text = "stubbed"
        self.next_finish = "endTurn"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self.next_text,
            tool_calls=[],
            finish_reason=self.next_finish,
            usage={},
            response_metadata=None,
        )


def _dummy_factory() -> Any:
    @contextlib.asynccontextmanager
    async def factory():
        yield None

    return factory


# ---------------------------------------------------------------------------
# LLMProviderSamplingHandler
# ---------------------------------------------------------------------------


def test_handler_requires_default_model() -> None:
    with pytest.raises(ValueError, match="default_model required"):
        LLMProviderSamplingHandler(_StubProvider(), default_model="")


def test_handler_requires_positive_token_cap() -> None:
    with pytest.raises(ValueError, match="max_tokens_cap"):
        LLMProviderSamplingHandler(
            _StubProvider(), default_model="m", max_tokens_cap=0,
        )


async def test_handler_uses_default_model_when_no_hints() -> None:
    p = _StubProvider()
    h = LLMProviderSamplingHandler(p, default_model="claude-x")
    result = await h(SamplingRequest(
        messages=[SamplingMessage(role="user", content="hi")],
    ))
    assert result.model == "claude-x"
    assert p.calls[0].model == "claude-x"


async def test_handler_picks_first_matching_hint() -> None:
    p = _StubProvider()
    h = LLMProviderSamplingHandler(
        p, default_model="default-m",
        allowed_models=["m1", "m2"],
    )
    # Hints: m3 (not allowed), m2 (allowed) → pick m2
    result = await h(SamplingRequest(
        messages=[SamplingMessage(role="user", content="x")],
        model_hints=["m3", "m2"],
    ))
    assert result.model == "m2"


async def test_handler_falls_back_to_default_when_no_hint_in_allowlist() -> None:
    p = _StubProvider()
    h = LLMProviderSamplingHandler(
        p, default_model="default-m",
        allowed_models=["m1"],
    )
    result = await h(SamplingRequest(
        messages=[SamplingMessage(role="user", content="x")],
        model_hints=["banned-1", "banned-2"],
    ))
    assert result.model == "default-m"


async def test_handler_caps_max_tokens_at_min_of_request_and_cap() -> None:
    p = _StubProvider()
    h = LLMProviderSamplingHandler(
        p, default_model="m", max_tokens_cap=1000,
    )
    # server requests 5000, cap is 1000 → 1000 wins
    await h(SamplingRequest(
        messages=[SamplingMessage(role="user", content="x")],
        max_tokens=5000,
    ))
    assert p.calls[0].max_output_tokens == 1000

    # server requests 500, cap is 1000 → 500 wins
    p.calls.clear()
    await h(SamplingRequest(
        messages=[SamplingMessage(role="user", content="x")],
        max_tokens=500,
    ))
    assert p.calls[0].max_output_tokens == 500


async def test_handler_uses_cap_when_request_omits_max_tokens() -> None:
    p = _StubProvider()
    h = LLMProviderSamplingHandler(p, default_model="m", max_tokens_cap=2048)
    await h(SamplingRequest(
        messages=[SamplingMessage(role="user", content="x")],
    ))
    assert p.calls[0].max_output_tokens == 2048


async def test_handler_prepends_system_prompt() -> None:
    from topsport_agent.types.message import Role

    p = _StubProvider()
    h = LLMProviderSamplingHandler(p, default_model="m")
    await h(SamplingRequest(
        messages=[SamplingMessage(role="user", content="hi")],
        system_prompt="You are a stub",
    ))
    msgs = p.calls[0].messages
    assert msgs[0].role == Role.SYSTEM
    assert msgs[0].content == "You are a stub"
    assert msgs[1].role == Role.USER


async def test_handler_returns_assistant_role_with_provider_text() -> None:
    p = _StubProvider()
    p.next_text = "stub reply"
    p.next_finish = "endTurn"
    h = LLMProviderSamplingHandler(p, default_model="m")
    result = await h(SamplingRequest(
        messages=[SamplingMessage(role="user", content="x")],
    ))
    assert result.role == "assistant"
    assert result.content == "stub reply"
    assert result.stop_reason == "endTurn"


# ---------------------------------------------------------------------------
# SDK boundary helpers
# ---------------------------------------------------------------------------


class _FakeTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeSDKMsg:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.content = _FakeTextContent(text)


class _FakeModelHint:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeModelPrefs:
    def __init__(self, hints: list[str], cost: float | None = None) -> None:
        self.hints = [_FakeModelHint(n) for n in hints]
        self.costPriority = cost
        self.speedPriority = None
        self.intelligencePriority = None


class _FakeSDKParams:
    def __init__(
        self, *, messages: list[Any], system: str | None = None,
        prefs: Any = None, max_tokens: int | None = None,
    ) -> None:
        self.messages = messages
        self.systemPrompt = system
        self.modelPreferences = prefs
        self.maxTokens = max_tokens
        self.temperature = None
        self.stopSequences = None


def test_from_sdk_params_basic() -> None:
    params = _FakeSDKParams(
        messages=[_FakeSDKMsg("user", "hi"), _FakeSDKMsg("assistant", "ok")],
        system="be brief",
        prefs=_FakeModelPrefs(["m1", "m2"], cost=0.5),
        max_tokens=512,
    )
    req = from_sdk_params(params)
    assert req.messages == [
        SamplingMessage(role="user", content="hi"),
        SamplingMessage(role="assistant", content="ok"),
    ]
    assert req.system_prompt == "be brief"
    assert req.model_hints == ["m1", "m2"]
    assert req.cost_priority == 0.5
    assert req.max_tokens == 512


def test_from_sdk_params_rejects_non_text_content() -> None:
    class _ImageContent:
        text = None  # SDK image content has no .text

    class _Msg:
        role = "user"
        content = _ImageContent()

    params = _FakeSDKParams(messages=[_Msg()])
    with pytest.raises(ValueError, match="non-text"):
        from_sdk_params(params)


def test_to_sdk_result_and_error_round_trip_via_real_sdk() -> None:
    pytest.importorskip("mcp")
    result = to_sdk_result(SamplingResult(
        role="assistant", content="hi", model="m", stop_reason="endTurn",
    ))
    from mcp.types import CreateMessageResult, ErrorData
    assert isinstance(result, CreateMessageResult)
    assert result.model == "m"

    err = to_sdk_error("oops", code=-32603)
    assert isinstance(err, ErrorData)
    assert err.code == -32603
    assert err.message == "oops"


# ---------------------------------------------------------------------------
# MCPClient sampling adapter
# ---------------------------------------------------------------------------


async def test_client_default_no_sampling_handler() -> None:
    client = MCPClient("s", _dummy_factory())
    assert client.sampling_handler is None


async def test_client_set_sampling_handler_persists() -> None:
    pytest.importorskip("mcp")
    p = _StubProvider()
    h = LLMProviderSamplingHandler(p, default_model="m")
    client = MCPClient("s", _dummy_factory())
    client.set_sampling_handler(h)
    assert client.sampling_handler is h
    client.set_sampling_handler(None)
    assert client.sampling_handler is None


async def test_sampling_callback_returns_method_not_found_when_no_handler() -> None:
    pytest.importorskip("mcp")
    client = MCPClient("s", _dummy_factory())
    result = await client._sampling_callback(_context=None, params=_FakeSDKParams(messages=[]))

    from mcp.types import ErrorData
    assert isinstance(result, ErrorData)
    assert result.code == -32601


async def test_sampling_callback_invalid_params_returns_error() -> None:
    pytest.importorskip("mcp")
    client = MCPClient("s", _dummy_factory())

    async def _h(req: SamplingRequest) -> SamplingResult:
        return SamplingResult(role="assistant", content="x", model="m")

    client.set_sampling_handler(_h)

    class _ImageContent:
        text = None

    class _Msg:
        role = "user"
        content = _ImageContent()

    params = _FakeSDKParams(messages=[_Msg()])
    result = await client._sampling_callback(_context=None, params=params)
    from mcp.types import ErrorData
    assert isinstance(result, ErrorData)
    assert result.code == -32602  # Invalid params


async def test_sampling_callback_handler_exception_returns_error(caplog) -> None:
    pytest.importorskip("mcp")
    client = MCPClient("name-x", _dummy_factory())

    async def _boom(req: SamplingRequest) -> SamplingResult:
        raise RuntimeError("handler boom")

    client.set_sampling_handler(_boom)
    params = _FakeSDKParams(messages=[_FakeSDKMsg("user", "x")])

    with caplog.at_level("WARNING", logger="topsport_agent.mcp.client"):
        result = await client._sampling_callback(_context=None, params=params)

    from mcp.types import ErrorData
    assert isinstance(result, ErrorData)
    assert result.code == -32603
    assert "handler boom" in result.message
    assert any("'name-x'" in r.message for r in caplog.records)


async def test_sampling_callback_returns_result_via_handler() -> None:
    pytest.importorskip("mcp")
    p = _StubProvider()
    p.next_text = "stub reply"
    h = LLMProviderSamplingHandler(p, default_model="m")
    client = MCPClient("s", _dummy_factory())
    client.set_sampling_handler(h)

    params = _FakeSDKParams(messages=[_FakeSDKMsg("user", "hi")])
    result = await client._sampling_callback(_context=None, params=params)

    from mcp.types import CreateMessageResult
    assert isinstance(result, CreateMessageResult)
    assert result.model == "m"


# ---------------------------------------------------------------------------
# Manager batch
# ---------------------------------------------------------------------------


async def test_manager_set_sampling_handler_applies_to_all() -> None:
    p = _StubProvider()
    h = LLMProviderSamplingHandler(p, default_model="m")
    manager = MCPManager()
    c1 = MCPClient("a", _dummy_factory())
    c2 = MCPClient("b", _dummy_factory())
    manager.register(c1)
    manager.register(c2)

    manager.set_sampling_handler(h)
    assert c1.sampling_handler is h
    assert c2.sampling_handler is h

    manager.set_sampling_handler(None)
    assert c1.sampling_handler is None
    assert c2.sampling_handler is None


# ---------------------------------------------------------------------------
# session_factory attach
# ---------------------------------------------------------------------------


def test_session_factory_attaches_sampling_callback_only_when_handler_set(
    monkeypatch,
) -> None:
    pytest.importorskip("mcp")
    from topsport_agent.mcp import client as client_mod

    captured: dict[str, Any] = {}

    class _FakeClientSession:
        def __init__(self, _r, _w, *, sampling_callback=None, **_):
            captured["sampling_callback"] = sampling_callback

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **kw):
            return None

        async def initialize(self):
            return None

    @contextlib.asynccontextmanager
    async def _fake_stdio_client(_p):
        yield (None, None)

    class _FakeStdio:
        def __init__(self, **kw):
            pass

    def _fake_import(name: str, *_):
        if name == "mcp":
            return type("M", (), {
                "ClientSession": _FakeClientSession,
                "StdioServerParameters": _FakeStdio,
            })
        if name == "mcp.client.stdio":
            return type("S", (), {"stdio_client": _fake_stdio_client})
        raise ImportError(name)

    monkeypatch.setattr(
        client_mod, "importlib",
        type("I", (), {"import_module": staticmethod(_fake_import)}),
    )

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    client = MCPClient.from_config(cfg)

    async def _run() -> None:
        async with client._session_factory():
            pass

    asyncio.run(_run())
    assert captured["sampling_callback"] is None

    p = _StubProvider()
    client.set_sampling_handler(LLMProviderSamplingHandler(p, default_model="m"))
    asyncio.run(_run())
    assert captured["sampling_callback"] == client._sampling_callback


# ---------------------------------------------------------------------------
# ServerConfig env + lifespan integration
# ---------------------------------------------------------------------------


def test_server_config_default_disables_sampling() -> None:
    cfg = ServerConfig()
    assert cfg.enable_mcp_sampling is False
    assert cfg.mcp_sampling_max_tokens == 4096


def test_server_config_from_env_reads_sampling_fields(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MCP_SAMPLING", "true")
    monkeypatch.setenv("MCP_SAMPLING_MAX_TOKENS", "1024")
    cfg = ServerConfig.from_env()
    assert cfg.enable_mcp_sampling is True
    assert cfg.mcp_sampling_max_tokens == 1024


def test_lifespan_fails_fast_when_sampling_enabled_but_model_empty() -> None:
    """ENABLE_MCP_SAMPLING=true 但 MODEL='' 时 lifespan 启动应抛 RuntimeError。"""
    cfg = ServerConfig(
        api_key="k",
        default_model="",  # 空 model
        auth_required=False,
        enable_brave_search=True,
        brave_api_key="k",
        enable_mcp_sampling=True,
    )
    p = _StubProvider()

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=p,
        agent_factory=factory,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="ENABLE_MCP_SAMPLING=true but MODEL is empty"):
        with TestClient(app):
            pass


# ---------------------------------------------------------------------------
# Rate limiting (P1 from review)
# ---------------------------------------------------------------------------


class _ManualClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


async def test_token_bucket_allows_within_capacity() -> None:
    bucket = TokenBucketRateLimit(capacity=3, refill_per_minute=60)
    await bucket.check("c")
    await bucket.check("c")
    await bucket.check("c")  # 3 calls in burst should pass


async def test_token_bucket_raises_when_exhausted() -> None:
    bucket = TokenBucketRateLimit(capacity=2, refill_per_minute=60)
    await bucket.check("c")
    await bucket.check("c")
    with pytest.raises(RateLimitExceeded, match="rate limit exceeded"):
        await bucket.check("c")


async def test_token_bucket_refills_over_time() -> None:
    clock = _ManualClock()
    # capacity=2, refill_per_minute=60 → 1 token/sec
    bucket = TokenBucketRateLimit(capacity=2, refill_per_minute=60, clock=clock)
    await bucket.check("c")
    await bucket.check("c")
    with pytest.raises(RateLimitExceeded):
        await bucket.check("c")

    clock.advance(1.5)  # 1.5 tokens refilled (capped at capacity=2)
    await bucket.check("c")  # OK now (0.5 token left after this)
    with pytest.raises(RateLimitExceeded):
        await bucket.check("c")  # 0.5 < 1


def test_token_bucket_rejects_invalid_init_values() -> None:
    with pytest.raises(ValueError, match="capacity"):
        TokenBucketRateLimit(capacity=0)
    with pytest.raises(ValueError, match="refill_per_minute"):
        TokenBucketRateLimit(capacity=10, refill_per_minute=0)


async def test_handler_propagates_rate_limit_exception() -> None:
    """Handler must raise (not swallow) RateLimitExceeded so _sampling_callback
    can transform it into JSON-RPC -32000 (server-defined rate limit)."""
    p = _StubProvider()
    bucket = TokenBucketRateLimit(capacity=1, refill_per_minute=60)
    h = LLMProviderSamplingHandler(
        p, default_model="m", rate_limit=bucket, client_name="x",
    )
    await h(SamplingRequest(messages=[SamplingMessage(role="user", content="hi")]))
    with pytest.raises(RateLimitExceeded):
        await h(SamplingRequest(messages=[SamplingMessage(role="user", content="hi")]))


async def test_sampling_callback_returns_rate_limit_code(caplog) -> None:
    """RateLimitExceeded → JSON-RPC -32000 (not the generic -32603)."""
    pytest.importorskip("mcp")
    p = _StubProvider()
    bucket = TokenBucketRateLimit(capacity=1, refill_per_minute=60)
    h = LLMProviderSamplingHandler(
        p, default_model="m", rate_limit=bucket, client_name="brave",
    )
    client = MCPClient("brave", _dummy_factory())
    client.set_sampling_handler(h)

    params = _FakeSDKParams(messages=[_FakeSDKMsg("user", "x")])
    # First call: OK
    result1 = await client._sampling_callback(_context=None, params=params)
    from mcp.types import CreateMessageResult, ErrorData
    assert isinstance(result1, CreateMessageResult)

    # Second call: rate limited
    with caplog.at_level("WARNING", logger="topsport_agent.mcp.client"):
        result2 = await client._sampling_callback(_context=None, params=params)
    assert isinstance(result2, ErrorData)
    assert result2.code == -32000
    assert "rate limit" in result2.message.lower()


async def test_handler_logs_audit_with_client_name(caplog) -> None:
    """Audit log must include client_name to enable post-hoc forensics."""
    p = _StubProvider()
    h = LLMProviderSamplingHandler(
        p, default_model="m", client_name="brave-search",
    )
    with caplog.at_level("INFO", logger="topsport_agent.mcp.sampling"):
        await h(SamplingRequest(
            messages=[SamplingMessage(role="user", content="x")],
        ))
    audit = [r for r in caplog.records if "client=" in r.message]
    assert audit
    assert "brave-search" in audit[0].message


async def test_handler_warns_when_stop_sequences_dropped(caplog) -> None:
    p = _StubProvider()
    h = LLMProviderSamplingHandler(p, default_model="m")
    with caplog.at_level("WARNING", logger="topsport_agent.mcp.sampling"):
        await h(SamplingRequest(
            messages=[SamplingMessage(role="user", content="x")],
            stop_sequences=["</done>"],
        ))
    assert any("stop_sequences" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Lifespan integration (after rate limit support landed)
# ---------------------------------------------------------------------------


def test_lifespan_warns_when_sampling_enabled_without_mcp_servers(
    caplog,
) -> None:
    """ENABLE_MCP_SAMPLING=true 但没配 MCP server → warning（非 fail-fast）。"""
    cfg = ServerConfig(
        api_key="k",
        default_model="m",
        auth_required=False,
        enable_mcp_sampling=True,
        enable_brave_search=False,  # 也没配置 mcp_config_path
    )
    p = _StubProvider()

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=p,
        agent_factory=factory,  # type: ignore[arg-type]
    )
    with caplog.at_level("WARNING", logger="topsport_agent.server.app"):
        with TestClient(app):
            pass
    assert any(
        "no MCP servers configured" in r.message for r in caplog.records
    )


def test_lifespan_attaches_sampling_handler_when_enabled() -> None:
    cfg = ServerConfig(
        api_key="k",
        default_model="m1",
        auth_required=False,
        enable_brave_search=True,
        brave_api_key="k",
        enable_mcp_sampling=True,
        mcp_sampling_max_tokens=2048,
    )
    p = _StubProvider()

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=p,
        agent_factory=factory,  # type: ignore[arg-type]
    )
    with TestClient(app):
        manager = app.state.mcp_manager
        assert manager is not None
        brave = manager.get("brave-search")
        assert brave is not None
        handler = brave.sampling_handler
        assert handler is not None
        # lifespan 应给每个 client 装配带 client_name + rate_limit 的独立实例
        assert isinstance(handler, LLMProviderSamplingHandler)
        assert handler._client_name == "brave-search"
        assert handler._rate_limit is not None
        assert isinstance(handler._rate_limit, TokenBucketRateLimit)
