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


def _make_test_app(provider: Any) -> Any:
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

    return create_app(
        ServerConfig(api_key="dummy", default_model="mock/test"),
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
        entry = store._entries["sess-fixed"]
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
        assert set(store._entries) == {"a", "b"}
        assert store._entries["a"].session.id == "a"
        assert store._entries["b"].session.id == "b"


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
        ServerConfig(api_key="dummy", max_sessions=2),
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
        # max=2，最早的 'a' 应被 evict
        assert "a" not in store._entries
        assert {"b", "c"}.issubset(store._entries.keys())


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
