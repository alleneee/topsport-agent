from __future__ import annotations

from ...types.message import Message

CHARS_PER_TOKEN = 4


def estimate_tokens(messages: list[Message]) -> int:
    total_chars = 0
    for msg in messages:
        if msg.content:
            total_chars += len(msg.content)
        for call in msg.tool_calls:
            total_chars += len(call.name) + len(str(call.arguments))
        for result in msg.tool_results:
            total_chars += len(str(result.output))
    return total_chars // CHARS_PER_TOKEN
