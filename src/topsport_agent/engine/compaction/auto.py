from __future__ import annotations

from typing import Any

from ...llm.request import LLMRequest
from ...llm.response import LLMResponse
from ...types.message import Message, Role
from .token_counter import estimate_tokens

SUMMARY_PROMPT = (
    "Summarize the following conversation concisely in 3-5 sentences. "
    "Focus on: what tasks were attempted, what tools were used, "
    "what results were obtained, and what decisions were made.\n\n"
)


async def auto_compact(
    messages: list[Message],
    *,
    session_goal: str | None,
    system_identity: str | None,
    provider: Any,
    summary_model: str,
    context_window: int,
    threshold: float,
    keep_recent: int,
) -> tuple[list[Message], bool]:
    token_count = estimate_tokens(messages)
    if token_count < int(context_window * threshold):
        return list(messages), False

    if len(messages) <= keep_recent:
        return list(messages), False

    old = messages[:-keep_recent]
    recent = messages[-keep_recent:]

    summary_text = await _summarize(old, provider, summary_model)

    reinject_parts: list[str] = []
    if system_identity:
        reinject_parts.append(f"## Identity\n\n{system_identity}")
    if session_goal:
        reinject_parts.append(f"## Task goal\n\n{session_goal}")
    reinject_parts.append(f"## Previous conversation summary\n\n{summary_text}")

    reinject_msg = Message(
        role=Role.SYSTEM,
        content="\n\n".join(reinject_parts),
    )

    return [reinject_msg] + list(recent), True


async def _summarize(messages: list[Message], provider: Any, model: str) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.role.value
        if msg.content:
            lines.append(f"[{role}] {msg.content[:500]}")
        for call in msg.tool_calls:
            lines.append(f"[{role}] tool_call: {call.name}({call.arguments})")
        for result in msg.tool_results:
            output_str = str(result.output)[:200]
            lines.append(f"[{role}] tool_result: {output_str}")

    conversation = "\n".join(lines)

    request = LLMRequest(
        model=model,
        messages=[
            Message(role=Role.USER, content=SUMMARY_PROMPT + conversation),
        ],
    )

    try:
        response: LLMResponse = await provider.complete(request)
        return response.text or "(summary unavailable)"
    except Exception:
        return "(summary generation failed)"
