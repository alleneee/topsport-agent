"""/v1/images/generations: OpenAI-compatible image generation endpoint.

Wraps OpenAIImageGenerationClient from llm/image_generation.py. The client is
built once at lifespan and kept on app.state.image_client. Absent -> 503.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..llm.image_generation import ImageGenerationRequest
from .auth import require_principal

_logger = logging.getLogger(__name__)

router = APIRouter()


class ImageGenRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=10)
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    response_format: Literal["url", "b64_json"] = "url"


@router.post("/v1/images/generations")
async def images_generations(
    body: ImageGenRequest,
    request: Request,
    principal: str = Depends(require_principal),
) -> dict:
    client = getattr(request.app.state, "image_client", None)
    if client is None:
        raise HTTPException(
            503,
            detail=(
                "image generation is not enabled on this server. "
                "set ENABLE_IMAGE_GEN=true and IMAGE_GEN_MODEL=<default model>."
            ),
        )
    default_model: str | None = getattr(request.app.state, "image_default_model", None)
    resolved_model = body.model or default_model
    if not resolved_model:
        raise HTTPException(
            400,
            detail="model is required (no default configured via IMAGE_GEN_MODEL)",
        )
    req = ImageGenerationRequest(
        prompt=body.prompt,
        model=resolved_model,
        size=body.size,
        quality=body.quality,
        style=body.style,
        n=body.n,
        response_format=body.response_format,
    )
    try:
        resp = await client.generate(req)
    except Exception as exc:
        _logger.exception(
            "image generation failed",
            extra={
                "event": "image_generation_failed",
                "principal": principal,
                "model": resolved_model,
            },
        )
        raise HTTPException(502, detail=f"image generation upstream error: {exc}") from exc

    data = []
    for img in resp.images:
        item: dict = {}
        if img.url is not None:
            item["url"] = img.url
        if img.b64_json is not None:
            item["b64_json"] = img.b64_json
        if img.revised_prompt is not None:
            item["revised_prompt"] = img.revised_prompt
        data.append(item)
    return {
        "id": f"imggen-{uuid.uuid4().hex[:20]}",
        "created": int(time.time()),
        "model": resolved_model,
        "data": data,
        "usage": resp.usage,
    }
