from __future__ import annotations

import json
from typing import Any

from ...types.message import Message, Role, ToolCall
from ...types.tool import ToolSpec
from ..request import LLMRequest
from ..response import (
    AssistantResponseBlock,
    LLMResponse,
    ProviderResponseMetadata,
)


class AnthropicMessagesAdapter:
    provider_name = "anthropic"

    def __init__(
        self,
        *,
        max_tokens: int = 4096,
        thinking_budget: int | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        options = dict(request.provider_options.get("anthropic", {}))
        system, converted_messages = self._convert_messages(request.messages)

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": int(
                request.max_output_tokens
                if request.max_output_tokens is not None
                else options.pop("max_tokens", self._max_tokens)
            ),
            "messages": converted_messages,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)

        thinking = options.pop("thinking", None)
        if thinking is None and self._thinking_budget:
            thinking = {
                "type": "enabled",
                "budget_tokens": int(self._thinking_budget),
            }
        if thinking:
            payload["thinking"] = thinking

        if request.temperature is not None:
            payload.setdefault("temperature", request.temperature)
        for key, value in options.items():
            payload.setdefault(key, value)
        return payload

    def parse_response(self, response: Any) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        assistant_blocks: list[AssistantResponseBlock] = []

        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "thinking":
                thinking = getattr(block, "thinking", None)
                if thinking:
                    thinking_block: AssistantResponseBlock = {
                        "type": "thinking",
                        "thinking": thinking,
                    }
                    signature = getattr(block, "signature", None)
                    if signature:
                        thinking_block["signature"] = signature
                    assistant_blocks.append(thinking_block)
            elif block_type == "text":
                text = getattr(block, "text", None)
                if text:
                    text_parts.append(text)
                    assistant_blocks.append({"type": "text", "text": text})
            elif block_type == "tool_use":
                block_payload = {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": dict(getattr(block, "input", None) or {}),
                }
                assistant_blocks.append(block_payload)
                tool_calls.append(
                    ToolCall(
                        id=block_payload["id"],
                        name=block_payload["name"],
                        arguments=dict(block_payload["input"]),
                    )
                )

        usage: dict[str, int] = {}
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            input_tokens = getattr(usage_obj, "input_tokens", None)
            output_tokens = getattr(usage_obj, "output_tokens", None)
            if input_tokens is not None:
                usage["input_tokens"] = int(input_tokens)
            if output_tokens is not None:
                usage["output_tokens"] = int(output_tokens)

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            finish_reason=getattr(response, "stop_reason", None) or "stop",
            usage=usage,
            response_metadata=ProviderResponseMetadata(
                provider=self.provider_name,
                assistant_blocks=assistant_blocks,
            ),
            raw=response,
        )

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush() -> None:
            if pending_tool_results:
                converted.append(
                    {"role": "user", "content": list(pending_tool_results)}
                )
                pending_tool_results.clear()

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_parts.append(msg.content)
                continue

            if msg.role == Role.TOOL:
                for result in msg.tool_results:
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": self._coerce_tool_result_content(result.output),
                    }
                    if result.is_error:
                        block["is_error"] = True
                    pending_tool_results.append(block)
                continue

            flush()

            if msg.role == Role.USER:
                converted.append({"role": "user", "content": msg.content or ""})
                continue

            if msg.role == Role.ASSISTANT:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for call in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                if content_blocks:
                    converted.append(
                        {"role": "assistant", "content": content_blocks}
                    )

        flush()

        system = "\n\n".join(system_parts) if system_parts else None
        return system, converted

    @staticmethod
    def _convert_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters or {"type": "object"},
            }
            for tool in tools
        ]

    @staticmethod
    def _coerce_tool_result_content(output: Any) -> list[dict[str, Any]]:
        if isinstance(output, str):
            return [{"type": "text", "text": output}]
        try:
            text = json.dumps(output, default=str, ensure_ascii=False)
        except Exception:
            text = str(output)
        return [{"type": "text", "text": text}]
