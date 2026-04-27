from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from topsport_agent.agent.base import Agent, AgentConfig, AgentRuntime
from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.image_generation import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from topsport_agent.types.message import (
    ContentPart,
    Message,
    Role,
    TextPart,
    image_url,
)


class _StubProvider:
    name = "stub"

    async def complete(self, request: Any) -> Any:
        return SimpleNamespace(
            text="ok",
            tool_calls=[],
            finish_reason="stop",
            usage={},
            response_metadata=None,
            raw=None,
        )


def _make_agent() -> Agent:
    provider = _StubProvider()
    engine = Engine(
        provider=provider,
        tools=[],
        config=EngineConfig(model="test-model"),
    )
    return Agent(provider=provider, config=AgentConfig(), engine=engine)


@pytest.mark.asyncio
async def test_run_accepts_string_as_before() -> None:
    agent = _make_agent()
    session = agent.new_session()
    async for _ in agent.run("hello", session):
        break
    assert session.messages[0].role == Role.USER
    assert session.messages[0].content == "hello"
    assert session.messages[0].content_parts is None


@pytest.mark.asyncio
async def test_run_accepts_content_parts_list() -> None:
    agent = _make_agent()
    session = agent.new_session()
    parts: list[ContentPart] = [
        TextPart("describe"),
        image_url("https://example.com/a.jpg"),
    ]
    async for _ in agent.run(parts, session):
        break
    msg = session.messages[0]
    assert msg.role == Role.USER
    assert msg.content is None
    assert msg.content_parts == parts


@pytest.mark.asyncio
async def test_run_accepts_prebuilt_message() -> None:
    agent = _make_agent()
    session = agent.new_session()
    msg = Message(
        role=Role.USER,
        content_parts=[TextPart("hi")],
        extra={"uid": "u1"},
    )
    async for _ in agent.run(msg, session):
        break
    assert session.messages[0] is msg
    assert session.messages[0].extra == {"uid": "u1"}

class _RecordingImageGen:
    """Test double for OpenAIImageGenerationClient."""

    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model
        self.last_request: ImageGenerationRequest | None = None

    async def generate(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        self.last_request = request
        return ImageGenerationResponse(
            images=[GeneratedImage(url="https://example.com/out.png")],
        )


def _make_agent_with_gen(gen: _RecordingImageGen) -> Agent:
    provider = _StubProvider()
    engine = Engine(
        provider=provider,
        tools=[],
        config=EngineConfig(model="test-model"),
    )
    return Agent(
        provider=provider,
        config=AgentConfig(),
        engine=engine,
        runtime=AgentRuntime(image_generator=gen),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_generate_image_raises_when_not_configured() -> None:
    agent = _make_agent()
    with pytest.raises(RuntimeError, match="No image_generator"):
        await agent.generate_image("anything")


@pytest.mark.asyncio
async def test_generate_image_delegates_and_uses_default_model() -> None:
    gen = _RecordingImageGen(default_model="dall-e-3")
    agent = _make_agent_with_gen(gen)
    resp = await agent.generate_image("cat", size="512x512")
    assert gen.last_request is not None
    assert gen.last_request.prompt == "cat"
    assert gen.last_request.size == "512x512"
    assert gen.last_request.model == "dall-e-3"
    assert resp.images[0].url == "https://example.com/out.png"


@pytest.mark.asyncio
async def test_generate_image_explicit_model_overrides_default() -> None:
    gen = _RecordingImageGen(default_model="dall-e-3")
    agent = _make_agent_with_gen(gen)
    await agent.generate_image("cat", model="gpt-image-1")
    assert gen.last_request is not None
    assert gen.last_request.model == "gpt-image-1"


@pytest.mark.asyncio
async def test_generate_image_requires_model_when_no_default() -> None:
    gen = _RecordingImageGen(default_model=None)
    agent = _make_agent_with_gen(gen)
    with pytest.raises(ValueError, match="model required"):
        await agent.generate_image("cat")


@pytest.mark.asyncio
async def test_from_config_accepts_runtime_image_generator() -> None:
    gen = _RecordingImageGen(default_model="dall-e-3")
    provider = _StubProvider()
    agent = Agent.from_config(
        provider=provider,
        config=AgentConfig(),
        runtime=AgentRuntime(image_generator=gen),  # type: ignore[arg-type]
    )  # type: ignore[arg-type]
    resp = await agent.generate_image("cat")
    assert resp.images[0].url == "https://example.com/out.png"
    assert gen.last_request is not None
    assert gen.last_request.model == "dall-e-3"
