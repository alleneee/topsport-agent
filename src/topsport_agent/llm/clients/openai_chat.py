from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any


class OpenAIChatClient:
    """OpenAI SDK 的薄封装：指数退避重试 + 可注入 sdk_client 以支持测试。与 Anthropic client 结构对称，方便统一维护。"""

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
        """指数退避重试，逻辑与 AnthropicMessagesClient.create 一致。"""
        attempt = 0
        while True:
            try:
                return await self._client.chat.completions.create(**payload)
            except Exception as exc:
                if attempt >= self._max_retries or not self._should_retry(exc):
                    raise
                await asyncio.sleep(self._retry_base_delay * (2**attempt))
                attempt += 1

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """OpenAI 流式 API 封装。

        产出事件：
          {"type": "text_delta", "text": "..."}
          {"type": "final_completion", "completion": <aggregated SimpleNamespace>}

        OpenAI 流式没有 get_final_message，这里手工累积 text / tool_calls / usage，
        构造与非流式 response 结构等价的 SimpleNamespace，供 adapter.parse_response 使用。
        """
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        # 请求返回 usage 统计，否则流式默认不带
        opts = dict(stream_payload.get("stream_options") or {})
        opts.setdefault("include_usage", True)
        stream_payload["stream_options"] = opts

        text_parts: list[str] = []
        # tool_calls 按 index 累积 arguments 字符串
        tc_buffer: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage_data: dict[str, int] = {}
        model_name: str | None = None

        stream = await self._client.chat.completions.create(**stream_payload)
        async for chunk in stream:
            if getattr(chunk, "model", None):
                model_name = chunk.model

            usage = getattr(chunk, "usage", None)
            if usage is not None:
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    val = getattr(usage, field, None)
                    if val is not None:
                        usage_data[field] = int(val)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            fr = getattr(choice, "finish_reason", None)
            if fr:
                finish_reason = fr

            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                yield {"type": "text_delta", "text": content}

            for tc_delta in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc_delta, "index", 0)
                entry = tc_buffer.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if getattr(tc_delta, "id", None):
                    entry["id"] = tc_delta.id
                func = getattr(tc_delta, "function", None)
                if func is not None:
                    if getattr(func, "name", None):
                        entry["name"] = func.name
                    if getattr(func, "arguments", None):
                        entry["arguments"] += func.arguments

        # 聚合为 non-streaming response 结构供 adapter 复用
        aggregated_message = SimpleNamespace(
            role="assistant",
            content="".join(text_parts) if text_parts else None,
            tool_calls=[
                SimpleNamespace(
                    id=entry["id"] or f"call_{idx}",
                    type="function",
                    function=SimpleNamespace(
                        name=entry["name"],
                        arguments=entry["arguments"],
                    ),
                )
                for idx, entry in sorted(tc_buffer.items())
            ],
        )
        aggregated_choice = SimpleNamespace(
            message=aggregated_message,
            finish_reason=finish_reason or "stop",
        )
        aggregated_completion = SimpleNamespace(
            choices=[aggregated_choice],
            usage=SimpleNamespace(**usage_data) if usage_data else None,
            model=model_name or payload.get("model", ""),
        )
        yield {"type": "final_completion", "completion": aggregated_completion}

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504, 529}:
            return True
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
