from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import AsyncIterator
from typing import Any


class AnthropicMessagesClient:
    """Anthropic SDK 的薄封装：指数退避重试 + 可注入 sdk_client 以支持测试。

    module 参数通过变量间接 importlib.import_module 绕过 Pyright reportMissingImports。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
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

        anthropic_module = module
        if anthropic_module is None:
            module_name = "anthropic"
            try:
                anthropic_module = importlib.import_module(module_name)
            except ImportError as exc:
                raise ImportError(
                    "anthropic is not installed. Run: uv sync --group llm"
                ) from exc

        kwargs: dict[str, Any] = {}
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if resolved_key:
            kwargs["api_key"] = resolved_key
        resolved_base = base_url or os.getenv("ANTHROPIC_BASE_URL")
        if resolved_base:
            kwargs["base_url"] = resolved_base

        self._client = anthropic_module.AsyncAnthropic(**kwargs)

    async def create(self, payload: dict[str, Any]) -> Any:
        """指数退避重试：仅对瞬态错误重试，永久性错误（如 401/403）直接抛出。"""
        attempt = 0
        while True:
            try:
                return await self._client.messages.create(**payload)
            except Exception as exc:
                if attempt >= self._max_retries or not self._should_retry(exc):
                    raise
                await asyncio.sleep(self._retry_base_delay * (2**attempt))
                attempt += 1

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Anthropic 流式 API 封装。

        产出抽象事件 dict：
          {"type": "text_delta", "text": "..."}
          {"type": "final_message", "message": <sdk message object>}

        重试策略（修 H-R1 流式/非流式不对称）：
        - 首个 delta yield 之前发生的瞬态错误做指数退避重试
        - 已经 yield 过内容后的错误原样抛出（无法回滚已交付的字节）
        """
        attempt = 0
        while True:
            yielded_any = False
            try:
                async with self._client.messages.stream(**payload) as stream:
                    async for text_chunk in stream.text_stream:
                        if text_chunk:
                            yield {"type": "text_delta", "text": text_chunk}
                            yielded_any = True
                    final_message = await stream.get_final_message()
                    yield {"type": "final_message", "message": final_message}
                return
            except Exception as exc:
                if yielded_any:
                    # 已经有内容流出，重试会破坏下游上下文 —— 原样抛出
                    raise
                if attempt >= self._max_retries or not self._should_retry(exc):
                    raise
                await asyncio.sleep(self._retry_base_delay * (2**attempt))
                attempt += 1

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        """529 是 Anthropic 特有的过载状态码，和标准 429/5xx 一起构成可重试集合。"""
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504, 529}:
            return True
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
