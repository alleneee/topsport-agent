from __future__ import annotations

import asyncio
import importlib
import os
from typing import Any


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        module: Any | None = None,
        sdk_client: Any | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        if sdk_client is not None:
            self._client = sdk_client
            return

        openai_module = module
        if openai_module is None:
            module_name = "openai"
            try:
                openai_module = importlib.import_module(module_name)
            except ImportError as exc:
                raise ImportError(
                    "openai is not installed. Run: uv sync --group llm"
                ) from exc

        kwargs: dict[str, Any] = {}
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if resolved_key:
            kwargs["api_key"] = resolved_key
        resolved_base = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_base:
            kwargs["base_url"] = resolved_base
        resolved_org = organization or os.getenv("OPENAI_ORGANIZATION")
        if resolved_org:
            kwargs["organization"] = resolved_org

        self._client = openai_module.AsyncOpenAI(**kwargs)

    async def create(self, payload: dict[str, Any]) -> Any:
        attempt = 0
        while True:
            try:
                return await self._client.chat.completions.create(**payload)
            except Exception as exc:
                if attempt >= self._max_retries or not self._should_retry(exc):
                    raise
                await asyncio.sleep(self._retry_base_delay * (2**attempt))
                attempt += 1

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504, 529}:
            return True
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
