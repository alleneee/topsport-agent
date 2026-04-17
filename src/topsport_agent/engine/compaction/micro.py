from __future__ import annotations

from ...types.message import Message, Role, ToolResult

PLACEHOLDER = "(previous output omitted)"


def micro_compact(
    messages: list[Message], keep_recent_tools: int = 3
) -> list[Message]:
    """只清空旧工具结果的 output，保留 call_id 和结构，维持 assistant->tool 消息配对不变量。"""
    tool_indices = [i for i, m in enumerate(messages) if m.role == Role.TOOL]
    if len(tool_indices) <= keep_recent_tools:
        return list(messages)

    # 分界线：最近 N 条工具消息保留完整内容，更早的替换为占位符。
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
