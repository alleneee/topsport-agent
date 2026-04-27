"""MCP `elicitation` capability tests.

覆盖：
- ElicitationRequest / Response 类型
- from_sdk_params: form 模式 / url 模式 / meta 透传
- to_sdk_result / to_sdk_error
- MCPClient.set_elicitation_handler 持久 + adapter 三种错误码
- contextvar 在 tool_bridge 进出 set/reset
- HTTPElicitationBroker:
    - handle 等 future 然后返回 response
    - handle 在无 active session_id 时立即 decline
    - handle 超时后 auto-decline
    - pending_for_session 列出未投递的请求并标记 delivered
    - resolve 找到 future 并 set_result（重复 resolve 返回 False）
- POST /v1/elicitations/<id> 端到端
- ServerConfig 字段 + lifespan 装配
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from starlette.testclient import TestClient

from topsport_agent.mcp.elicitation import (
    ElicitationRequest,
    ElicitationResponse,
    current_session_id,
    from_sdk_params,
    to_sdk_error,
    to_sdk_result,
)
from topsport_agent.mcp import MCPClient, MCPServerConfig
from topsport_agent.mcp.types import MCPTransport
from topsport_agent.server.app import create_app
from topsport_agent.server.config import ServerConfig
from topsport_agent.server.elicitation import HTTPElicitationBroker


def _dummy_factory() -> Any:
    @contextlib.asynccontextmanager
    async def factory():
        yield None

    return factory


# ---------------------------------------------------------------------------
# from_sdk_params: form / url
# ---------------------------------------------------------------------------


class _FakeSDKFormParams:
    mode = "form"
    message = "What's your favorite color?"
    requestedSchema = {"type": "object", "properties": {"color": {"type": "string"}}}
    meta = {"trace_id": "abc"}


class _FakeSDKUrlParams:
    mode = "url"
    message = "Click to authorize"
    url = "https://example.com/oauth"
    elicitationId = "server-side-eid"
    meta = None


def test_from_sdk_params_form_mode() -> None:
    req = from_sdk_params(_FakeSDKFormParams(), request_id="abc123")
    assert req.id == "abc123"
    assert req.mode == "form"
    assert req.message == "What's your favorite color?"
    assert req.requested_schema == {
        "type": "object",
        "properties": {"color": {"type": "string"}},
    }
    assert req.url is None
    assert req.meta == {"trace_id": "abc"}


def test_from_sdk_params_url_mode() -> None:
    req = from_sdk_params(_FakeSDKUrlParams(), request_id="xyz")
    assert req.mode == "url"
    assert req.url == "https://example.com/oauth"
    assert req.elicitation_id_url_mode == "server-side-eid"
    assert req.requested_schema is None


def test_to_sdk_result_round_trip() -> None:
    pytest.importorskip("mcp")
    sdk = to_sdk_result(ElicitationResponse(action="accept", content={"x": 1}))
    from mcp.types import ElicitResult
    assert isinstance(sdk, ElicitResult)
    assert sdk.action == "accept"
    assert sdk.content == {"x": 1}


def test_to_sdk_error_round_trip() -> None:
    pytest.importorskip("mcp")
    err = to_sdk_error("nope", code=-32601)
    from mcp.types import ErrorData
    assert isinstance(err, ErrorData)
    assert err.code == -32601


# ---------------------------------------------------------------------------
# MCPClient adapter
# ---------------------------------------------------------------------------


async def test_client_default_no_elicitation_handler() -> None:
    client = MCPClient("s", _dummy_factory())
    assert client.elicitation_handler is None


async def test_client_set_elicitation_handler_persists() -> None:
    client = MCPClient("s", _dummy_factory())

    async def h(req: ElicitationRequest) -> ElicitationResponse:
        return ElicitationResponse(action="decline")

    client.set_elicitation_handler(h)
    assert client.elicitation_handler is h
    client.set_elicitation_handler(None)
    assert client.elicitation_handler is None


async def test_elicitation_callback_method_not_found_when_no_handler() -> None:
    pytest.importorskip("mcp")
    client = MCPClient("s", _dummy_factory())
    result = await client._elicitation_callback(
        _context=None, params=_FakeSDKFormParams(),
    )
    from mcp.types import ErrorData
    assert isinstance(result, ErrorData)
    assert result.code == -32601


async def test_elicitation_callback_handler_returns_result() -> None:
    pytest.importorskip("mcp")
    client = MCPClient("s", _dummy_factory())

    async def h(req: ElicitationRequest) -> ElicitationResponse:
        return ElicitationResponse(action="accept", content={"answer": "blue"})

    client.set_elicitation_handler(h)
    result = await client._elicitation_callback(
        _context=None, params=_FakeSDKFormParams(),
    )
    from mcp.types import ElicitResult
    assert isinstance(result, ElicitResult)
    assert result.action == "accept"
    assert result.content == {"answer": "blue"}


async def test_elicitation_callback_handler_exception_returns_error() -> None:
    pytest.importorskip("mcp")
    client = MCPClient("name-x", _dummy_factory())

    async def boom(req: ElicitationRequest) -> ElicitationResponse:
        raise RuntimeError("handler exploded")

    client.set_elicitation_handler(boom)
    result = await client._elicitation_callback(
        _context=None, params=_FakeSDKFormParams(),
    )
    from mcp.types import ErrorData
    assert isinstance(result, ErrorData)
    assert result.code == -32603
    assert "handler exploded" in result.message


# ---------------------------------------------------------------------------
# HTTPElicitationBroker
# ---------------------------------------------------------------------------


async def test_broker_handle_with_no_active_session_cancels() -> None:
    """No session_id on request → auto-cancel (not decline; spec semantics
    distinguishes cancel=no-answer vs decline=refused)."""
    broker = HTTPElicitationBroker()
    req = ElicitationRequest(id="r1", message="?", mode="form", session_id=None)
    response = await broker.handle(req)
    assert response.action == "cancel"


async def test_broker_handle_routes_to_session_and_returns_response() -> None:
    broker = HTTPElicitationBroker()
    req = ElicitationRequest(
        id="r1", message="?", mode="form", session_id="session-A",
    )

    async def replier() -> None:
        await asyncio.sleep(0.05)
        ok = await broker.resolve(
            "r1", ElicitationResponse(action="accept", content={"v": 1}),
            expected_session_id="session-A",
        )
        assert ok

    replier_task = asyncio.create_task(replier())
    response = await broker.handle(req)
    await replier_task

    assert response.action == "accept"
    assert response.content == {"v": 1}


async def test_broker_handle_timeout_auto_cancels() -> None:
    """Timeout = no answer = cancel (retryable), NOT decline (refused)."""
    broker = HTTPElicitationBroker(default_timeout_seconds=0.1)
    req = ElicitationRequest(
        id="r1", message="?", mode="form", session_id="session-A",
    )
    response = await broker.handle(req)
    assert response.action == "cancel"


async def test_broker_pending_for_session_marks_delivered_once() -> None:
    broker = HTTPElicitationBroker()
    req = ElicitationRequest(
        id="r1", message="?", mode="form", session_id="session-A",
    )

    handle_task = asyncio.create_task(broker.handle(req))
    await asyncio.sleep(0.01)

    first = await broker.pending_for_session("session-A")
    assert len(first) == 1
    assert first[0][0] == "r1"
    assert first[0][1].message == "?"

    second = await broker.pending_for_session("session-A")
    assert second == []

    await broker.resolve(
        "r1", ElicitationResponse(action="cancel"),
        expected_session_id="session-A",
    )
    await handle_task


async def test_broker_pending_for_other_sessions_not_visible() -> None:
    broker = HTTPElicitationBroker()
    req = ElicitationRequest(
        id="r1", message="?", mode="form", session_id="session-A",
    )

    handle_task = asyncio.create_task(broker.handle(req))
    await asyncio.sleep(0.01)

    other = await broker.pending_for_session("session-B")
    assert other == []  # session-B 看不到 session-A 的请求

    await broker.resolve(
        "r1", ElicitationResponse(action="cancel"),
        expected_session_id="session-A",
    )
    await handle_task


async def test_broker_resolve_unknown_id_returns_false() -> None:
    broker = HTTPElicitationBroker()
    ok = await broker.resolve(
        "nonexistent", ElicitationResponse(action="accept"),
    )
    assert ok is False


async def test_broker_resolve_with_wrong_session_id_returns_false() -> None:
    """跨租户答复防护：用户 B 拿到 session A 的 elicitation_id 也无法答。"""
    broker = HTTPElicitationBroker()
    req = ElicitationRequest(
        id="r1", message="?", mode="form", session_id="session-A",
    )

    handle_task = asyncio.create_task(broker.handle(req))
    await asyncio.sleep(0.01)

    # session-B 试图答 session-A 的请求 → False（id-probing 防御：返回 False
    # 而非 raise，与"不存在"无差别）
    ok = await broker.resolve(
        "r1", ElicitationResponse(action="accept"),
        expected_session_id="session-B",
    )
    assert ok is False

    # 正确 session 仍能答
    ok2 = await broker.resolve(
        "r1", ElicitationResponse(action="cancel"),
        expected_session_id="session-A",
    )
    assert ok2
    await handle_task


async def test_broker_resolve_after_already_resolved_returns_false() -> None:
    broker = HTTPElicitationBroker()
    req = ElicitationRequest(
        id="r1", message="?", mode="form", session_id="session-A",
    )

    handle_task = asyncio.create_task(broker.handle(req))
    await asyncio.sleep(0.01)
    ok1 = await broker.resolve(
        "r1", ElicitationResponse(action="accept"),
        expected_session_id="session-A",
    )
    assert ok1
    await handle_task

    ok2 = await broker.resolve(
        "r1", ElicitationResponse(action="accept"),
        expected_session_id="session-A",
    )
    assert ok2 is False


async def test_broker_signal_for_session_lazily_creates_event() -> None:
    """First signal_for() call creates the Event; later calls return same."""
    broker = HTTPElicitationBroker()
    sig1 = broker.signal_for("session-A")
    sig2 = broker.signal_for("session-A")
    assert sig1 is sig2  # cached
    sig3 = broker.signal_for("session-B")
    assert sig3 is not sig1


async def test_broker_handle_sets_session_signal() -> None:
    """handle() must set the per-session asyncio.Event so listeners
    awaiting it wake up immediately when a new elicitation arrives."""
    broker = HTTPElicitationBroker(default_timeout_seconds=5.0)
    sig = broker.signal_for("session-A")
    assert not sig.is_set()

    req = ElicitationRequest(
        id="r1", message="?", mode="form", session_id="session-A",
    )
    handle_task = asyncio.create_task(broker.handle(req))
    # Wait for handle to register pending — small grace period
    await asyncio.sleep(0.01)

    assert sig.is_set(), "signal must be set after handle registers pending"

    # Cleanup
    await broker.resolve(
        "r1", ElicitationResponse(action="cancel"),
        expected_session_id="session-A",
    )
    await handle_task


async def test_broker_signal_does_not_fire_for_other_sessions() -> None:
    broker = HTTPElicitationBroker(default_timeout_seconds=5.0)
    sig_a = broker.signal_for("session-A")
    sig_b = broker.signal_for("session-B")

    req = ElicitationRequest(
        id="r1", message="?", mode="form", session_id="session-A",
    )
    handle_task = asyncio.create_task(broker.handle(req))
    await asyncio.sleep(0.01)

    assert sig_a.is_set()
    assert not sig_b.is_set(), "session-B signal must NOT fire on session-A elicit"

    await broker.resolve(
        "r1", ElicitationResponse(action="cancel"),
        expected_session_id="session-A",
    )
    await handle_task


def test_broker_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError):
        HTTPElicitationBroker(default_timeout_seconds=0)
    with pytest.raises(ValueError):
        HTTPElicitationBroker(default_timeout_seconds=-5)


# ---------------------------------------------------------------------------
# tool_bridge ContextVar
# ---------------------------------------------------------------------------


async def test_tool_bridge_sets_session_id_on_client_field_when_elicitation_enabled() -> None:
    """When elicitation_handler is set, MCPToolSource handler routes
    session_id via client._current_call_session_id (instance field, not
    ContextVar — ContextVar can't cross the SDK task boundary).
    """
    from topsport_agent.mcp.tool_bridge import MCPToolSource
    from topsport_agent.types.tool import ToolContext

    seen_sid: list[str | None] = []

    class _Sess:
        async def call_tool(self, name, **kwargs):
            seen_sid.append(client._current_call_session_id)

            class _R:
                content = []
                isError = False
                structuredContent = None

            return _R()

    @contextlib.asynccontextmanager
    async def factory():
        yield _Sess()

    client = MCPClient("c", factory)

    async def stub_handler(req: ElicitationRequest) -> ElicitationResponse:
        return ElicitationResponse(action="cancel")

    client.set_elicitation_handler(stub_handler)
    source = MCPToolSource(client)

    class _MockMCPTool:
        name = "echo"
        description = ""
        inputSchema = {"type": "object"}

    tool_spec = source._wrap(_MockMCPTool())

    assert client._current_call_session_id is None
    await tool_spec.handler(
        {},
        ToolContext(
            session_id="my-session-id", call_id="c1",
            cancel_event=asyncio.Event(),
        ),
    )
    assert seen_sid == ["my-session-id"]
    # Reset after call
    assert client._current_call_session_id is None


async def test_tool_bridge_skips_lock_when_elicitation_disabled() -> None:
    """No elicitation_handler → no lock overhead; concurrent call_tool
    runs concurrently (lock would serialise them)."""
    from topsport_agent.mcp.tool_bridge import MCPToolSource
    from topsport_agent.types.tool import ToolContext

    in_flight: dict[str, int] = {"count": 0, "max": 0}

    class _Sess:
        async def call_tool(self, name, **kwargs):
            in_flight["count"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["count"])
            await asyncio.sleep(0.02)
            in_flight["count"] -= 1

            class _R:
                content = []
                isError = False
                structuredContent = None

            return _R()

    @contextlib.asynccontextmanager
    async def factory():
        yield _Sess()

    client = MCPClient("c", factory)
    # No elicitation handler set → no lock
    source = MCPToolSource(client)

    class _MockMCPTool:
        name = "echo"
        description = ""
        inputSchema = {"type": "object"}

    tool_spec = source._wrap(_MockMCPTool())

    ctx = ToolContext(
        session_id="s", call_id="c", cancel_event=asyncio.Event(),
    )
    await asyncio.gather(*(tool_spec.handler({}, ctx) for _ in range(3)))
    assert in_flight["max"] >= 2, "calls should run concurrently without lock"


# ---------------------------------------------------------------------------
# ServerConfig env wiring + lifespan integration
# ---------------------------------------------------------------------------


def test_server_config_default_disables_elicitation() -> None:
    cfg = ServerConfig()
    assert cfg.enable_mcp_elicitation is False
    assert cfg.mcp_elicitation_timeout_seconds == 60.0


def test_server_config_from_env_reads_elicitation_fields(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_MCP_ELICITATION", "true")
    monkeypatch.setenv("MCP_ELICITATION_TIMEOUT_SECONDS", "30")
    cfg = ServerConfig.from_env()
    assert cfg.enable_mcp_elicitation is True
    assert cfg.mcp_elicitation_timeout_seconds == 30.0


def _stub_provider() -> Any:
    class _P:
        name = "stub"

        async def complete(self, request):
            from topsport_agent.llm.provider import LLMResponse
            return LLMResponse(text="ok", finish_reason="stop")

    return _P()


def test_lifespan_attaches_elicitation_handler_when_enabled() -> None:
    cfg = ServerConfig(
        api_key="k", default_model="m", auth_required=False,
        enable_brave_search=True, brave_api_key="k",
        enable_mcp_elicitation=True,
        mcp_elicitation_timeout_seconds=5.0,
    )

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=_stub_provider(),
        agent_factory=factory,  # type: ignore[arg-type]
    )
    with TestClient(app):
        broker = app.state.elicitation_broker
        assert broker is not None
        assert broker.default_timeout == 5.0

        manager = app.state.mcp_manager
        assert manager is not None
        brave = manager.get("brave-search")
        assert brave is not None
        assert brave.elicitation_handler is not None


def test_lifespan_no_broker_when_elicitation_disabled() -> None:
    cfg = ServerConfig(
        api_key="k", default_model="m", auth_required=False,
        enable_brave_search=True, brave_api_key="k",
        enable_mcp_elicitation=False,
    )

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=_stub_provider(),
        agent_factory=factory,  # type: ignore[arg-type]
    )
    with TestClient(app):
        assert app.state.elicitation_broker is None


# ---------------------------------------------------------------------------
# POST /v1/elicitations/<id> endpoint end-to-end
# ---------------------------------------------------------------------------


def test_post_elicitation_resolves_pending() -> None:
    cfg = ServerConfig(
        api_key="k", default_model="m", auth_required=False,
        enable_brave_search=True, brave_api_key="k",
        enable_mcp_elicitation=True,
    )

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=_stub_provider(),
        agent_factory=factory,  # type: ignore[arg-type]
    )

    with TestClient(app) as tc:
        broker: HTTPElicitationBroker = app.state.elicitation_broker

        async def _setup_and_test() -> dict[str, Any]:
            req = ElicitationRequest(
                id="eid-1", message="hello?", mode="form",
                session_id="session-A",
            )
            handle_task = asyncio.create_task(broker.handle(req))
            await asyncio.sleep(0.01)

            resp = tc.post(
                "/v1/elicitations/eid-1",
                headers={"X-Session-Id": "session-A"},
                json={"action": "accept", "content": {"answer": "world"}},
            )
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

            response = await handle_task
            return {
                "action": response.action,
                "content": response.content,
            }

        result = asyncio.run(_setup_and_test())
        assert result == {"action": "accept", "content": {"answer": "world"}}


def test_post_elicitation_missing_session_header_returns_400() -> None:
    cfg = ServerConfig(
        api_key="k", default_model="m", auth_required=False,
        enable_brave_search=True, brave_api_key="k",
        enable_mcp_elicitation=True,
    )

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=_stub_provider(),
        agent_factory=factory,  # type: ignore[arg-type]
    )

    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/elicitations/x",
            json={"action": "accept", "content": {}},
        )
        assert resp.status_code == 400
        assert "X-Session-Id" in resp.json()["detail"]


def test_post_elicitation_wrong_session_id_returns_404() -> None:
    """跨租户答复防护：用户 B 用自己的 session_id 答 session A 的 id → 404
    （不暴露该 id 是否存在于其他 session）。"""
    cfg = ServerConfig(
        api_key="k", default_model="m", auth_required=False,
        enable_brave_search=True, brave_api_key="k",
        enable_mcp_elicitation=True,
    )

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=_stub_provider(),
        agent_factory=factory,  # type: ignore[arg-type]
    )

    with TestClient(app) as tc:
        broker: HTTPElicitationBroker = app.state.elicitation_broker

        async def _setup() -> int:
            req = ElicitationRequest(
                id="eid-x", message="?", mode="form",
                session_id="session-A",
            )
            handle_task = asyncio.create_task(broker.handle(req))
            await asyncio.sleep(0.01)

            resp = tc.post(
                "/v1/elicitations/eid-x",
                headers={"X-Session-Id": "session-B"},
                json={"action": "accept"},
            )

            # Cleanup: cancel pending so test doesn't leak the task
            await broker.resolve(
                "eid-x", ElicitationResponse(action="cancel"),
                expected_session_id="session-A",
            )
            await handle_task
            return resp.status_code

        status = asyncio.run(_setup())
        assert status == 404


def test_post_elicitation_unknown_id_returns_404() -> None:
    cfg = ServerConfig(
        api_key="k", default_model="m", auth_required=False,
        enable_brave_search=True, brave_api_key="k",
        enable_mcp_elicitation=True,
    )

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=_stub_provider(),
        agent_factory=factory,  # type: ignore[arg-type]
    )

    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/elicitations/never-existed",
            headers={"X-Session-Id": "any-session"},
            json={"action": "accept", "content": {}},
        )
        assert resp.status_code == 404


def test_post_elicitation_when_broker_not_configured_returns_404() -> None:
    cfg = ServerConfig(
        api_key="k", default_model="m", auth_required=False,
        enable_mcp_elicitation=False,  # broker not built
    )

    def factory(_prov, _model):
        from topsport_agent.agent import default_agent
        return default_agent(_prov, _model)

    app = create_app(
        cfg, provider_name="anthropic", provider=_stub_provider(),
        agent_factory=factory,  # type: ignore[arg-type]
    )

    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/elicitations/x",
            headers={"X-Session-Id": "any"},
            json={"action": "accept", "content": {}},
        )
        assert resp.status_code == 404
        assert "broker not configured" in resp.json()["detail"].lower()
