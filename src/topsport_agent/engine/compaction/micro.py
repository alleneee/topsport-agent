from __future__ import annotations

from ...types.message import Message, Role, ToolResult

PLACEHOLDER = "(previous output omitted)"


def micro_compact(
    messages: list[Message], keep_recent_tools: int = 3
) -> list[Message]:
    tool_indices = [i for i, m in enumerate(messages) if m.role == Role.TOOL]
    if len(tool_indices) <= keep_recent_tools:
        return list(messages)

    cutoff = tool_indices[-keep_recent_tools]
    result: list[Message] = []
    for i, msg in enumerate(messages):
        if msg.role == Role.TOOL and i < cutoff:
            compressed = [
                ToolResult(
                    call_id=tr.call_id,
                    output=PLACEHOLDER,
                    is_error=tr.is_error,
                )
                for tr in msg.tool_results
            ]
            result.append(Message(role=Role.TOOL, tool_results=compressed))
        else:
            result.append(msg)
    return result
