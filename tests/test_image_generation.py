from __future__ import annotations

from typing import Any

import pytest

from topsport_agent.llm.image_generation import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResponse,
)


def test_image_generation_request_defaults() -> None:
    req = ImageGenerationRequest(prompt="a cat", model="dall-e-3")
    assert req.prompt == "a cat"
    assert req.model == "dall-e-3"
    assert req.size is None
    assert req.quality is None
    assert req.style is None
    assert req.n == 1
    assert req.response_format == "url"
    assert req.provider_options == {}


def test_image_generation_request_accepts_all_options() -> None:
    req = ImageGenerationRequest(
        prompt="cat",
        model="dall-e-3",
        size="1024x1024",
        quality="hd",
        style="vivid",
        n=2,
        response_format="b64_json",
        provider_options={"user": "u1"},
    )
    assert req.size == "1024x1024"
    assert req.quality == "hd"
    assert req.response_format == "b64_json"
    assert req.provider_options == {"user": "u1"}


def test_generated_image_defaults() -> None:
    img = GeneratedImage()
    assert img.url is None
    assert img.b64_json is None
    assert img.revised_prompt is None


def test_image_generation_response_holds_images() -> None:
    resp = ImageGenerationResponse(
        images=[GeneratedImage(url="https://x/out.png")],
    )
    assert len(resp.images) == 1
    assert resp.usage == {}


from types import SimpleNamespace

from topsport_agent.llm.image_generation import OpenAIImageGenerationClient


class _CapturingImages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.captured: dict[str, Any] | None = None

    async def generate(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        return self.result


class _MockClient:
    def __init__(self, result: Any) -> None:
        self.images = _CapturingImages(result)


def _result(url: str = "https://example.com/out.png") -> Any:
    return SimpleNamespace(
        data=[SimpleNamespace(url=url, b64_json=None, revised_prompt="refined")],
        usage=None,
    )


def test_client_requires_client_or_factory() -> None:
    with pytest.raises(ValueError, match="client.*factory"):
        OpenAIImageGenerationClient()


@pytest.mark.asyncio
async def test_generate_builds_kwargs_and_parses_response() -> None:
    mock = _MockClient(_result())
    client = OpenAIImageGenerationClient(client=mock)
    resp = await client.generate(
        ImageGenerationRequest(
            prompt="cat", model="dall-e-3", size="1024x1024", quality="hd",
        )
    )
    assert mock.images.captured == {
        "model": "dall-e-3",
        "prompt": "cat",
        "n": 1,
        "response_format": "url",
        "size": "1024x1024",
        "quality": "hd",
    }
    assert resp.images[0].url == "https://example.com/out.png"
    assert resp.images[0].revised_prompt == "refined"


@pytest.mark.asyncio
async def test_provider_options_override_kwargs() -> None:
    mock = _MockClient(_result())
    client = OpenAIImageGenerationClient(client=mock)
    await client.generate(
        ImageGenerationRequest(
            prompt="x", model="dall-e-3",
            provider_options={"model": "overridden", "user": "u1"},
        )
    )
    assert mock.images.captured is not None
    assert mock.images.captured["model"] == "overridden"
    assert mock.images.captured["user"] == "u1"


@pytest.mark.asyncio
async def test_lazy_factory_creates_client_once() -> None:
    calls = {"n": 0}

    def _factory() -> Any:
        calls["n"] += 1
        return _MockClient(_result())

    client = OpenAIImageGenerationClient(client_factory=_factory)
    await client.generate(ImageGenerationRequest(prompt="a", model="m"))
    await client.generate(ImageGenerationRequest(prompt="b", model="m"))
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_parse_response_handles_empty_data_list() -> None:
    mock = _MockClient(SimpleNamespace(data=[], usage=None))
    client = OpenAIImageGenerationClient(client=mock)
    resp = await client.generate(ImageGenerationRequest(prompt="x", model="m"))
    assert resp.images == []


@pytest.mark.asyncio
async def test_parse_response_extracts_usage_when_present() -> None:
    mock = _MockClient(
        SimpleNamespace(
            data=[SimpleNamespace(url="u", b64_json=None, revised_prompt=None)],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=20, total_tokens=30
            ),
        )
    )
    client = OpenAIImageGenerationClient(client=mock)
    resp = await client.generate(ImageGenerationRequest(prompt="x", model="m"))
    assert resp.usage == {
        "input_tokens": 10, "output_tokens": 20, "total_tokens": 30
    }


@pytest.mark.asyncio
async def test_save_b64_writes_decoded_bytes(tmp_path) -> None:
    import base64 as _b64
    payload = _b64.b64encode(b"fake-png-bytes").decode("ascii")
    img = GeneratedImage(b64_json=payload)
    target = await img.save(tmp_path / "out.png")
    assert target.read_bytes() == b"fake-png-bytes"


@pytest.mark.asyncio
async def test_save_url_uses_injected_http_client(tmp_path) -> None:
    class _MockResp:
        def __init__(self, content: bytes) -> None:
            self.content = content
        def raise_for_status(self) -> None:
            pass

    class _MockHttp:
        def __init__(self) -> None:
            self.calls: list[str] = []
        async def get(self, url: str) -> _MockResp:
            self.calls.append(url)
            return _MockResp(b"downloaded")

    http = _MockHttp()
    img = GeneratedImage(url="https://example.com/x.png")
    target = await img.save(tmp_path / "out.png", http_client=http)
    assert http.calls == ["https://example.com/x.png"]
    assert target.read_bytes() == b"downloaded"


@pytest.mark.asyncio
async def test_save_raises_when_neither_url_nor_b64(tmp_path) -> None:
    img = GeneratedImage()
    with pytest.raises(ValueError, match="neither url nor b64_json"):
        await img.save(tmp_path / "x.png")
