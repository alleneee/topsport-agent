from __future__ import annotations

import re
from typing import Any

from ...llm.request import LLMRequest
from ...llm.response import LLMResponse
from ...types.message import Message, Role
from ..prompt import PROTECTED_TAGS
from .token_counter import estimate_tokens

SUMMARY_PROMPT = (
    "Summarize the following conversation concisely in 3-5 sentences. "
    "Focus on: what tasks were attempted, what tools were used, "
    "what results were obtained, and what decisions were made.\n\n"
)

# 匹配 <tag>...</tag> 区块的正则（非贪婪，跨行）
_SECTION_RE = re.compile(r"<([a-z][a-z0-9-]*)>\n(.*?)\n</\1>", re.DOTALL)


def compact_system_prompt(
    system_text: str,
    *,
    drop_tags: frozenset[str] | None = None,
) -> str:
    """从系统提示词中移除指定标签的区块。

    用于在 token 预算紧张时选择性丢弃低价值 section，
    保留 PROTECTED_TAGS 中的关键区块。
    """
    if drop_tags is None:
        return system_text

    def _replace(match: re.Match[str]) -> str:
        tag = match.group(1)
        if tag in drop_tags and tag not in PROTECTED_TAGS:
            return ""
        return match.group(0)

    result = _SECTION_RE.sub(_replace, system_text)
    # 清理多余空行
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def extract_protected_sections(system_text: str) -> str:
    """只保留 PROTECTED_TAGS 标签内的区块，其余全部丢弃。

    用于极端压缩场景：保留 system-prompt / identity / instructions，
    丢弃所有记忆、技能、插件上下文。
    """
    parts: list[str] = []
    for match in _SECTION_RE.finditer(system_text):
        tag = match.group(1)
        if tag in PROTECTED_TAGS:
            parts.append(match.group(0))
    return "\n\n".join(parts) if parts else system_text


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
    """超过上下文窗口阈值时，用 LLM 摘要替换旧消息，同时重注入身份和目标防止压缩后"失忆"。"""
    token_count = estimate_tokens(messages)
    if token_count < int(context_window * threshold):
        return list(messages), False

    if len(messages) <= keep_recent:
        return list(messages), False

    old = messages[:-keep_recent]
    recent = messages[-keep_recent:]

    # 摘要失败时宁可保留原始消息，不丢数据。
    try:
        summary_text = await _summarize(old, provider, summary_model)
    except Exception:
        return list(messages), False

    # 压缩后 LLM 丢失了身份和任务上下文，必须在摘要消息中重新注入。
    reinject_parts: list[str] = []
    if system_identity:
        reinject_parts.append(f"<identity>\n{system_identity}\n</identity>")
    if session_goal:
        reinject_parts.append(f"<session-state>\n## Task goal\n\n{session_goal}\n</session-state>")
    reinject_parts.append(f"<context>\n## Previous conversation summary\n\n{summary_text}\n</context>")

    reinject_msg = Message(
        role=Role.SYSTEM,
        content="\n\n".join(reinject_parts),
    )

    return [reinject_msg] + list(recent), True


async def _summarize(messages: list[Message], provider: Any, model: str) -> str:
    """将消息序列展平为纯文本，截断单条内容防止摘要请求本身超长。"""
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

    response: LLMResponse = await provider.complete(request)
    return response.text or "(summary unavailable)"
