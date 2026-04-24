from __future__ import annotations

import base64
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


@dataclass(slots=True)
class ImageGenerationRequest:
    """Image generation request aligned with OpenAI /v1/images/generations."""
    prompt: str
    model: str
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    n: int = 1
    response_format: Literal["url", "b64_json"] = "url"
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class GeneratedImage:
    """Single generated image. Exactly one of url/b64_json is set by the
    provider; revised_prompt is DALL-E-3-specific."""
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None

    async def save(
        self,
        path: str | Path,
        *,
        http_client: Any | None = None,
    ) -> Path:
        """Save this image to disk. b64_json is decoded; url is downloaded."""
        target = Path(path)
        if self.b64_json is not None:
            target.write_bytes(base64.b64decode(self.b64_json))
            return target
        if self.url is not None:
            mod_name = "httpx"
            httpx_mod = importlib.import_module(mod_name)
            if http_client is None:
                async with httpx_mod.AsyncClient() as c:
                    resp = await c.get(self.url)
                    resp.raise_for_status()
                    target.write_bytes(resp.content)
            else:
                resp = await http_client.get(self.url)
                resp.raise_for_status()
                target.write_bytes(resp.content)
            return target
        raise ValueError("GeneratedImage has neither url nor b64_json")


@dataclass(slots=True)
class ImageGenerationResponse:
    """Response from OpenAIImageGenerationClient.generate."""
    images: list[GeneratedImage]
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class OpenAIImageGenerationClient:
    """Synchronous-request client for OpenAI /v1/images/generations.

    Construct with an injected `client` (for tests) or a `client_factory`
    (for production, lazy-imports openai.AsyncOpenAI).
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        default_model: str | None = None,
    ) -> None:
        if client is None and client_factory is None:
            raise ValueError(
                "Either `client` or `client_factory` is required"
            )
        self._client = client
        self._client_factory = client_factory
        self.default_model = default_model

    async def generate(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        client = self._ensure_client()
        kwargs = self._build_kwargs(request)
        raw = await client.images.generate(**kwargs)
        return self._parse(raw)

    def _ensure_client(self) -> Any:
        if self._client is None:
            assert self._client_factory is not None
            self._client = self._client_factory()
        return self._client

    @staticmethod
    def _build_kwargs(req: ImageGenerationRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": req.model,
            "prompt": req.prompt,
            "n": req.n,
            "response_format": req.response_format,
        }
        if req.size:
            kwargs["size"] = req.size
        if req.quality:
            kwargs["quality"] = req.quality
        if req.style:
            kwargs["style"] = req.style
        kwargs.update(req.provider_options)
        return kwargs

    @staticmethod
    def _parse(raw: Any) -> ImageGenerationResponse:
        data = getattr(raw, "data", None) or []
        images = [
            GeneratedImage(
                url=getattr(item, "url", None),
                b64_json=getattr(item, "b64_json", None),
                revised_prompt=getattr(item, "revised_prompt", None),
            )
            for item in data
        ]
        usage: dict[str, int] = {}
        usage_obj = getattr(raw, "usage", None)
        if usage_obj is not None:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                val = getattr(usage_obj, key, None)
                if val is not None:
                    usage[key] = int(val)
        return ImageGenerationResponse(images=images, usage=usage, raw=raw)
