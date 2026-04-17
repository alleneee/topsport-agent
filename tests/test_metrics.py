"""Prometheus metrics subscriber 单元 + server 集成测试。"""

from __future__ import annotations

import pytest

prometheus_client = pytest.importorskip("prometheus_client")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from topsport_agent.observability import PrometheusMetrics  # noqa: E402
from topsport_agent.types.events import Event, EventType  # noqa: E402


def _fresh_metrics() -> PrometheusMetrics:
    """每个测试用独立 registry，互不污染。"""
    return PrometheusMetrics(registry=prometheus_client.CollectorRegistry())


async def test_metrics_counts_steps_and_runs() -> None:
    m = _fresh_metrics()
    await m.on_event(Event(type=EventType.STEP_START, session_id="s"))
    await m.on_event(Event(type=EventType.STEP_START, session_id="s"))
    await m.on_event(
        Event(type=EventType.RUN_END, session_id="s", payload={"final_state": "done"})
    )

    text, _ = m.render()
    body = text.decode()
    assert "topsport_agent_steps_total 2.0" in body
    assert 'topsport_agent_runs_total{state="done"} 1.0' in body


async def test_metrics_records_tokens_both_conventions() -> None:
    m = _fresh_metrics()
    # Anthropic 字段
    await m.on_event(
        Event(
            type=EventType.LLM_CALL_END,
            session_id="s",
            payload={"usage": {"input_tokens": 100, "output_tokens": 50}},
        )
    )
    # OpenAI 字段
    await m.on_event(
        Event(
            type=EventType.LLM_CALL_END,
            session_id="s",
            payload={"usage": {"prompt_tokens": 30, "completion_tokens": 20}},
        )
    )

    body = m.render()[0].decode()
    assert 'topsport_agent_llm_tokens_total{direction="prompt"} 130.0' in body
    assert 'topsport_agent_llm_tokens_total{direction="completion"} 70.0' in body


async def test_metrics_tool_calls_labelled() -> None:
    m = _fresh_metrics()
    await m.on_event(
        Event(
            type=EventType.TOOL_CALL_END,
            session_id="s",
            payload={"name": "read_file", "is_error": False, "call_id": "c1"},
        )
    )
    await m.on_event(
        Event(
            type=EventType.TOOL_CALL_END,
            session_id="s",
            payload={"name": "write_file", "is_error": True, "call_id": "c2"},
        )
    )

    body = m.render()[0].decode()
    assert 'topsport_agent_tool_calls_total{is_error="0",name="read_file"} 1.0' in body
    assert 'topsport_agent_tool_calls_total{is_error="1",name="write_file"} 1.0' in body


async def test_metrics_llm_duration_histogram_captures() -> None:
    m = _fresh_metrics()
    await m.on_event(Event(type=EventType.LLM_CALL_START, session_id="s"))
    await m.on_event(
        Event(
            type=EventType.LLM_CALL_END,
            session_id="s",
            payload={"usage": {}},
        )
    )
    body = m.render()[0].decode()
    assert "topsport_agent_llm_call_duration_seconds_count 1.0" in body


def test_metrics_endpoint_served_when_metrics_provided() -> None:
    from dataclasses import dataclass, field
    from collections.abc import AsyncIterator
    from topsport_agent.agent.base import Agent, AgentConfig
    from topsport_agent.llm.provider import LLMProvider
    from topsport_agent.llm.request import LLMRequest
    from topsport_agent.llm.response import LLMResponse
    from topsport_agent.llm.stream import LLMStreamChunk
    from topsport_agent.server import ServerConfig, create_app

    @dataclass
    class _P:
        name: str = "mock"
        calls: list[LLMRequest] = field(default_factory=list)

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls.append(request)
            return LLMResponse(text="ok", tool_calls=[], finish_reason="stop")

        async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
            yield LLMStreamChunk(type="text_delta", text_delta="ok")
            yield LLMStreamChunk(
                type="done",
                final_response=LLMResponse(text="ok", finish_reason="stop"),
            )

    def _factory(p: LLMProvider, model: str) -> Agent:
        return Agent.from_config(
            p,
            AgentConfig(
                name="t", description="", system_prompt="", model=model,
                enable_skills=False, enable_memory=False, enable_plugins=False,
                enable_browser=False, stream=True,
            ),
        )

    m = _fresh_metrics()
    app = create_app(
        ServerConfig(api_key="dummy", auth_required=False),
        provider_name="anthropic",
        provider=_P(),
        agent_factory=_factory,
        metrics=m,
    )
    with TestClient(app) as client:
        # 发一次 chat 让 metrics 有内容可看
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/m",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200

        r = client.get("/metrics")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        assert "topsport_agent_steps_total" in r.text


def test_metrics_missing_endpoint_when_not_provided() -> None:
    from dataclasses import dataclass, field
    from collections.abc import AsyncIterator
    from topsport_agent.agent.base import Agent, AgentConfig
    from topsport_agent.llm.provider import LLMProvider
    from topsport_agent.llm.request import LLMRequest
    from topsport_agent.llm.response import LLMResponse
    from topsport_agent.llm.stream import LLMStreamChunk
    from topsport_agent.server import ServerConfig, create_app

    @dataclass
    class _P:
        name: str = "mock"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(text="ok", finish_reason="stop")

        async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
            yield LLMStreamChunk(type="done",
                                 final_response=LLMResponse(text="ok", finish_reason="stop"))

    def _factory(p: LLMProvider, model: str) -> Agent:
        return Agent.from_config(
            p,
            AgentConfig(
                name="t", description="", system_prompt="", model=model,
                enable_skills=False, enable_memory=False, enable_plugins=False,
                enable_browser=False, stream=True,
            ),
        )

    app = create_app(
        ServerConfig(api_key="dummy", auth_required=False),
        provider_name="anthropic",
        provider=_P(),
        agent_factory=_factory,
    )
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 404
