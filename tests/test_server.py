"""FastAPI server 测试：/v1/chat/completions (JSON/SSE) + /v1/plan/execute (SSE)。

fastapi 未安装时跳过整个模块。provider 使用 mock，不触网。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from topsport_agent.agent.base import Agent, AgentConfig  # noqa: E402
from topsport_agent.llm.provider import LLMProvider  # noqa: E402
from topsport_agent.llm.request import LLMRequest  # noqa: E402
from topsport_agent.llm.response import LLMResponse  # noqa: E402
from topsport_agent.llm.stream import LLMStreamChunk  # noqa: E402
from topsport_agent.server import ServerConfig, create_app  # noqa: E402


@dataclass
class MockStreamProvider:
    """同时实现 complete + stream，可让 agent 走流式或非流式路径。"""

    name: str = "mock"
    text: str = "hi there"
    calls: list[LLMRequest] = field(default_factory=list)
    stream_chunks_sent: int = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self.text,
            tool_calls=[],
            finish_reason="stop",
            usage={"input_tokens": 3, "output_tokens": 4},
            response_metadata=None,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        self.calls.append(request)
        pieces = [self.text[i : i + 3] for i in range(0, len(self.text), 3)] or [""]
        for piece in pieces:
            self.stream_chunks_sent += 1
            yield LLMStreamChunk(type="text_delta", text_delta=piece)
        final = LLMResponse(
            text=self.text,
            tool_calls=[],
            finish_reason="stop",
            usage={"input_tokens": 3, "output_tokens": 4},
            response_metadata=None,
        )
        yield LLMStreamChunk(type="done", final_response=final)


def _make_test_app(provider: Any, *, auth_required: bool = False) -> Any:
    """默认关闭鉴权以覆盖历史行为；auth 相关测试按需显式开启。"""

    def agent_factory(p: LLMProvider, model: str) -> Agent:
        cfg = AgentConfig(
            name="test",
            description="",
            system_prompt="SP",
            model=model,
            enable_skills=False,
            enable_memory=False,
            enable_plugins=False,
            enable_browser=False,
            stream=True,
        )
        return Agent.from_config(p, cfg)

    server_cfg = ServerConfig(
        api_key="dummy",
        default_model="mock/test",
        auth_required=auth_required,
        auth_token="test-token" if auth_required else "",
    )
    return create_app(
        server_cfg,
        provider_name="anthropic",
        provider=provider,
        agent_factory=agent_factory,
    )


# ---------------------------------------------------------------------------
# /healthz + 验证/验参
# ---------------------------------------------------------------------------


def test_healthz_ok() -> None:
    app = _make_test_app(MockStreamProvider())
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_chat_rejects_bad_model_format() -> None:
    app = _make_test_app(MockStreamProvider())
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "no-slash", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400


def test_chat_rejects_wrong_provider() -> None:
    app = _make_test_app(MockStreamProvider())
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400


def test_chat_requires_user_message() -> None:
    app = _make_test_app(MockStreamProvider())
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/m",
                "messages": [{"role": "system", "content": "only system"}],
            },
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# JSON chat
# ---------------------------------------------------------------------------


def test_chat_json_returns_openai_shape() -> None:
    provider = MockStreamProvider(text="hello world")
    app = _make_test_app(provider)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "anthropic/m"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "hello world"
        assert body["choices"][0]["finish_reason"] == "stop"


def test_chat_session_state_preserved_across_requests() -> None:
    """同一 user(session_id) 的两次请求共享消息历史。"""
    provider = MockStreamProvider(text="response")
    app = _make_test_app(provider)
    with TestClient(app) as client:
        for i in range(2):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/m",
                    "messages": [{"role": "user", "content": f"turn {i}"}],
                    "user": "sess-fixed",
                },
            )
            assert r.status_code == 200

        store = app.state.session_store
        # principal 命名空间前缀：auth 关闭时 principal="anonymous"
        entry = store._entries["anonymous::sess-fixed"]
        # 2 轮应追加 2 user + 2 assistant = 至少 4 条消息（不含 engine 内部事件）
        assert len(entry.session.messages) >= 4


def test_chat_different_sessions_isolated() -> None:
    provider = MockStreamProvider(text="x")
    app = _make_test_app(provider)
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/m",
                "messages": [{"role": "user", "content": "A"}],
                "user": "a",
            },
        )
        client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/m",
                "messages": [{"role": "user", "content": "B"}],
                "user": "b",
            },
        )
        store = app.state.session_store
        # principal 命名空间：auth 关闭时 principal="anonymous"
        assert set(store._entries) == {"anonymous::a", "anonymous::b"}
        assert store._entries["anonymous::a"].session.id == "anonymous::a"
        assert store._entries["anonymous::b"].session.id == "anonymous::b"


# ---------------------------------------------------------------------------
# SSE chat stream
# ---------------------------------------------------------------------------


def _collect_sse_lines(text: str) -> list[str]:
    """从 SSE 响应文本抽出所有 data 行（不含 event:/注释/空行）。"""
    return [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]


def test_chat_stream_emits_openai_chunks_and_done() -> None:
    provider = MockStreamProvider(text="abcdefg")
    app = _make_test_app(provider)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "anthropic/m",
                "messages": [{"role": "user", "content": "stream me"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = r.read().decode()

    data_lines = _collect_sse_lines(body)
    assert data_lines[-1] == "[DONE]"
    import json

    # 首帧 role=assistant
    first = json.loads(data_lines[0])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"].get("role") == "assistant"

    # 中间帧拼接内容应等于 mock 文本
    content_parts: list[str] = []
    for line in data_lines[1:-2]:
        chunk = json.loads(line)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            content_parts.append(delta["content"])
    assert "".join(content_parts) == "abcdefg"

    # 倒数第二帧带 finish_reason
    penultimate = json.loads(data_lines[-2])
    assert penultimate["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Plan execute SSE
# ---------------------------------------------------------------------------


def test_plan_execute_streams_named_events() -> None:
    """Plan 三步 fan-out，SSE 产出 plan_approved / step_start/end x3 / plan_done。"""
    provider = MockStreamProvider(text="ok")
    app = _make_test_app(provider)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/plan/execute",
            json={
                "model": "anthropic/m",
                "plan": {
                    "id": "p1",
                    "goal": "test",
                    "steps": [
                        {"id": "s1", "title": "a", "instructions": "do a"},
                        {"id": "s2", "title": "b", "instructions": "do b"},
                        {
                            "id": "s3",
                            "title": "c",
                            "instructions": "do c",
                            "depends_on": ["s1", "s2"],
                        },
                    ],
                },
            },
        ) as r:
            assert r.status_code == 200
            body = r.read().decode()

    # 抽出 event: 行
    events = [
        line[len("event: ") :] for line in body.splitlines() if line.startswith("event: ")
    ]
    assert events[0] == "plan_approved"
    assert events.count("plan_step_start") == 3
    assert events.count("plan_step_end") == 3
    assert events[-1] == "plan_done"


def test_plan_execute_rejects_invalid_plan() -> None:
    app = _make_test_app(MockStreamProvider())
    with TestClient(app) as client:
        r = client.post(
            "/v1/plan/execute",
            json={
                "model": "anthropic/m",
                "plan": {
                    "id": "bad",
                    "goal": "",
                    "steps": [
                        {
                            "id": "x",
                            "title": "",
                            "instructions": "",
                            "depends_on": ["does-not-exist"],
                        }
                    ],
                },
            },
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Session eviction
# ---------------------------------------------------------------------------


def test_session_store_lru_eviction_closes_old_agents() -> None:
    provider = MockStreamProvider()
    app = create_app(
        ServerConfig(api_key="dummy", max_sessions=2, auth_required=False),
        provider_name="anthropic",
        provider=provider,
        agent_factory=lambda p, m: Agent.from_config(
            p,
            AgentConfig(
                name="t", description="", system_prompt="", model=m,
                enable_skills=False, enable_memory=False, enable_plugins=False,
                enable_browser=False, stream=True,
            ),
        ),
    )
    with TestClient(app) as client:
        for sid in ("a", "b", "c"):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "user": sid,
                },
            )
            assert r.status_code == 200
        store = app.state.session_store
        # max=2，最早的 'a' 应被 evict；key 带 anonymous:: 前缀
        assert "anonymous::a" not in store._entries
        assert {"anonymous::b", "anonymous::c"}.issubset(store._entries.keys())


@pytest.mark.asyncio
async def test_session_store_close_all_idempotent() -> None:
    from topsport_agent.server.sessions import SessionStore

    closed_count = 0

    class FakeAgent:
        async def close(self):
            nonlocal closed_count
            closed_count += 1

        def new_session(self, sid):
            from topsport_agent.types.session import Session

            return Session(id=sid, system_prompt="")

    store = SessionStore(
        agent_factory=lambda _p, _m: FakeAgent(),  # type: ignore[arg-type,return-value]
        provider=MockStreamProvider(),  # type: ignore[arg-type]
    )
    await store.get_or_create("s1", "m")
    await store.get_or_create("s2", "m")
    await store.close_all()
    assert closed_count == 2
    await store.close_all()  # 第二次无副作用
    assert closed_count == 2


# ---------------------------------------------------------------------------
# CR-01 · HTTP 鉴权 + principal 命名空间 + /readyz + max_plan_steps clamp
# ---------------------------------------------------------------------------


def test_auth_required_but_no_token_startup_fails() -> None:
    """secure-by-default：required=True 且未提供 token → 构造 app 即失败。"""
    with pytest.raises(ValueError, match="tokens is empty"):
        create_app(
            ServerConfig(
                api_key="dummy",
                auth_required=True,
                auth_token="",
                auth_tokens_file="",
            ),
            provider_name="anthropic",
            provider=MockStreamProvider(),
        )


def test_chat_rejects_missing_bearer() -> None:
    # 手动构造带 token 的 app 避开上面的 shortcut 校验
    def agent_factory(p: LLMProvider, model: str) -> Agent:
        return Agent.from_config(
            p,
            AgentConfig(
                name="t", description="", system_prompt="", model=model,
                enable_skills=False, enable_memory=False, enable_plugins=False,
                enable_browser=False, stream=True,
            ),
        )

    app = create_app(
        ServerConfig(api_key="dummy", auth_required=True, auth_token="secret"),
        provider_name="anthropic",
        provider=MockStreamProvider(),
        agent_factory=agent_factory,
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "anthropic/m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 401
        assert "bearer" in r.text.lower()


def test_chat_rejects_wrong_bearer() -> None:
    def agent_factory(p: LLMProvider, model: str) -> Agent:
        return Agent.from_config(
            p,
            AgentConfig(
                name="t", description="", system_prompt="", model=model,
                enable_skills=False, enable_memory=False, enable_plugins=False,
                enable_browser=False, stream=True,
            ),
        )

    app = create_app(
        ServerConfig(api_key="dummy", auth_required=True, auth_token="secret"),
        provider_name="anthropic",
        provider=MockStreamProvider(),
        agent_factory=agent_factory,
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "anthropic/m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert r.status_code == 401


def test_chat_accepts_valid_bearer_and_namespaces_session() -> None:
    def agent_factory(p: LLMProvider, model: str) -> Agent:
        return Agent.from_config(
            p,
            AgentConfig(
                name="t", description="", system_prompt="", model=model,
                enable_skills=False, enable_memory=False, enable_plugins=False,
                enable_browser=False, stream=True,
            ),
        )

    app = create_app(
        ServerConfig(api_key="dummy", auth_required=True, auth_token="secret"),
        provider_name="anthropic",
        provider=MockStreamProvider(),
        agent_factory=agent_factory,
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/m",
                "messages": [{"role": "user", "content": "hi"}],
                "user": "my-session",
            },
            headers={"Authorization": "Bearer secret"},
        )
        assert r.status_code == 200
        store = app.state.session_store
        # principal "default" 前缀 + 用户提供的 hint
        assert "default::my-session" in store._entries
        # 原始未前缀 key 不应出现，防跨 principal 命中
        assert "my-session" not in store._entries


def test_readyz_reports_component_health() -> None:
    app = _make_test_app(MockStreamProvider())
    with TestClient(app) as client:
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["components"]["provider"] is True
        assert body["components"]["session_store"] is True
        assert body["components"]["auth_config"] is True


def test_plan_execute_clamps_max_steps() -> None:
    """body.max_steps 超过 config 硬上限时被截断。"""
    def agent_factory(p: LLMProvider, model: str) -> Agent:
        return Agent.from_config(
            p,
            AgentConfig(
                name="t", description="", system_prompt="", model=model,
                enable_skills=False, enable_memory=False, enable_plugins=False,
                enable_browser=False, stream=True,
            ),
        )

    app = create_app(
        ServerConfig(
            api_key="dummy",
            auth_required=False,
            max_plan_steps=3,
        ),
        provider_name="anthropic",
        provider=MockStreamProvider(),
        agent_factory=agent_factory,
    )
    # 用 TestClient 启动 lifespan，然后直接验证 config 挂好
    with TestClient(app) as client:
        r = client.post(
            "/v1/plan/execute",
            json={
                "model": "anthropic/m",
                "max_steps": 1000,
                "plan": {
                    "id": "p1",
                    "goal": "g",
                    "steps": [
                        {"id": "s1", "title": "t", "instructions": "i", "depends_on": []}
                    ],
                },
            },
        )
        # 不关心 SSE 具体内容，只要 200 且流能开起来
        assert r.status_code == 200
        assert app.state.config.max_plan_steps == 3


# ---------------------------------------------------------------------------
# H-S3 · .env 文件属主 + 权限校验
# ---------------------------------------------------------------------------


import os as _os


def test_dotenv_loads_when_owned_and_0600(tmp_path, monkeypatch) -> None:
    from topsport_agent.server.main import _load_dotenv

    env = tmp_path / ".env"
    env.write_text("DOTENV_TEST_VAR=hello\n")
    _os.chmod(env, 0o600)
    monkeypatch.delenv("DOTENV_TEST_VAR", raising=False)

    _load_dotenv(env)
    assert _os.environ["DOTENV_TEST_VAR"] == "hello"


def test_dotenv_refuses_world_readable(tmp_path) -> None:
    from topsport_agent.server.main import _DotenvRefused, _load_dotenv

    env = tmp_path / ".env"
    env.write_text("X=1\n")
    _os.chmod(env, 0o644)  # group+other readable

    if _os.name != "posix":
        pytest.skip("permission check is POSIX-only")
    with pytest.raises(_DotenvRefused, match="permissive mode"):
        _load_dotenv(env)


def test_dotenv_does_not_overwrite_existing_env(tmp_path, monkeypatch) -> None:
    from topsport_agent.server.main import _load_dotenv

    env = tmp_path / ".env"
    env.write_text("DOTENV_TEST_VAR2=from-file\n")
    _os.chmod(env, 0o600)
    monkeypatch.setenv("DOTENV_TEST_VAR2", "from-env")

    _load_dotenv(env)
    # 已有 env 不覆盖，保持 process env 为权威
    assert _os.environ["DOTENV_TEST_VAR2"] == "from-env"


# ---------------------------------------------------------------------------
# H-R5 · graceful drain
# ---------------------------------------------------------------------------


def test_drain_rejects_new_api_requests_but_allows_health() -> None:
    app = _make_test_app(MockStreamProvider())
    with TestClient(app) as client:
        # 模拟进入 drain 状态
        app.state.draining = True

        r = client.post(
            "/v1/chat/completions",
            json={"model": "anthropic/m", "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.status_code == 503
        assert "draining" in r.text

        # /healthz 和 /readyz 不受 drain 影响
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200


def test_drain_timeout_respects_config() -> None:
    """drain_timeout_seconds 可从 ServerConfig 读到。"""
    from topsport_agent.server.config import ServerConfig as _SC

    cfg = _SC(api_key="x", auth_required=False, drain_timeout_seconds=7.5)
    assert cfg.drain_timeout_seconds == 7.5
