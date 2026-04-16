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


class OpenAIChatAdapter:
    provider_name = "openai"

    def __init__(
        self,
        *,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        options = dict(request.provider_options.get("openai", {}))
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._convert_messages(request.messages),
        }

        max_tokens_key = (
            "max_completion_tokens"
            if "max_completion_tokens" in options
            else "max_tokens"
        )
        default_max_tokens = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else self._max_tokens
        )
        payload[max_tokens_key] = int(
            options.pop(max_tokens_key, default_max_tokens)
        )

        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)

        reasoning_effort = options.pop("reasoning_effort", self._reasoning_effort)
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        if request.temperature is not None:
            payload.setdefault("temperature", request.temperature)
        if request.tool_choice is not None:
            payload.setdefault("tool_choice", request.tool_choice)
        for key, value in options.items():
            payload.setdefault(key, value)
        return payload

    def parse_response(self, completion: Any) -> LLMResponse:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return LLMResponse(text=None, raw=completion)

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            return LLMResponse(text=None, raw=completion)

        text = getattr(message, "content", None)
        assistant_blocks: list[AssistantResponseBlock] = []
        if text:
            assistant_blocks.append({"type": "text", "text": text})
        tool_calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            function = getattr(raw_call, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", "")
            raw_arguments = getattr(function, "arguments", "") or ""
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": raw_arguments}
            tool_calls.append(
                ToolCall(
                    id=getattr(raw_call, "id", ""),
                    name=name,
                    arguments=(
                        dict(arguments)
                        if isinstance(arguments, dict)
                        else {"_value": arguments}
                    ),
                )
            )
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": getattr(raw_call, "id", ""),
                    "name": name,
                    "input": (
                        dict(arguments)
                        if isinstance(arguments, dict)
                        else {"_value": arguments}
                    ),
                    "raw_arguments": raw_arguments,
                }
            )

        usage: dict[str, int] = {}
        usage_obj = getattr(completion, "usage", None)
        if usage_obj is not None:
            prompt = getattr(usage_obj, "prompt_tokens", None)
            completion_tokens = getattr(usage_obj, "completion_tokens", None)
            if prompt is not None:
                usage["input_tokens"] = int(prompt)
            if completion_tokens is not None:
                usage["output_tokens"] = int(completion_tokens)

        finish_reason = getattr(choice, "finish_reason", None) or "stop"
        response_metadata = ProviderResponseMetadata(
            provider=self.provider_name,
            assistant_blocks=assistant_blocks,
        )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            response_metadata=response_metadata,
            raw=completion,
        )

    def _convert_messages(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    converted.append({"role": "system", "content": msg.content})
                continue

            if msg.role == Role.USER:
                converted.append({"role": "user", "content": msg.content or ""})
                continue

            if msg.role == Role.ASSISTANT:
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content,
                }
                if msg.tool_calls:
                    assistant["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(
                                    call.arguments, ensure_ascii=False
                                ),
                            },
                        }
                        for call in msg.tool_calls
                    ]
                converted.append(assistant)
                continue

            if msg.role == Role.TOOL:
                for result in msg.tool_results:
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": self._coerce_output(result.output),
                        }
                    )

        return converted

    @staticmethod
    def _convert_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters or {"type": "object"},
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _coerce_output(output: Any) -> str:
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, default=str, ensure_ascii=False)
        except Exception:
            return str(output)
