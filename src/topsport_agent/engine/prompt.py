"""Prompt Section 注册与组装机制。

参考 Claude Code 的标签结构，将 system prompt 划分为多个语义区块，
每个区块用 XML 标签包裹，按优先级排序。好处：
1. LLM 能明确区分不同来源的上下文（记忆 / 技能 / 插件 / 指令）
2. 压缩模块可按标签选择性压缩低价值区块，保留关键任务上下文
3. 调试时可清晰看到 prompt 的组成结构
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SectionPriority(IntEnum):
    """内置 section 优先级。自定义 section 可使用任意 int 值。"""

    SYSTEM_PROMPT = 0
    IDENTITY = 100
    WORKING_MEMORY = 200
    SKILLS_CATALOG = 300
    ACTIVE_SKILLS = 400
    PLUGIN_CONTEXT = 500
    TOOLS_GUIDE = 600
    SESSION_STATE = 700
    INSTRUCTIONS = 900


# 压缩时默认保护的标签 — 这些区块不会被选择性压缩
PROTECTED_TAGS = frozenset({
    "system-prompt",
    "identity",
    "instructions",
})

# 压缩时可安全缩减的标签 — 按优先级从低到高压缩
COMPRESSIBLE_TAGS = (
    "session-state",
    "tools-guide",
    "plugin-context",
    "skills-catalog",
    "working-memory",
)


@dataclass(slots=True)
class PromptSection:
    """系统提示词的一个语义区块。"""

    tag: str
    priority: int
    content: str


class PromptBuilder:
    """收集多个 section，按优先级排序，用 XML 标签包裹后组装为完整 system prompt。

    用法：
        builder = PromptBuilder()
        builder.add("system-prompt", "You are a helpful assistant.", 0)
        builder.add("working-memory", "[goal] Refactor pipeline", 200)
        text = builder.build()
    """

    def __init__(self) -> None:
        self._sections: list[PromptSection] = []

    def add(self, tag: str, content: str, priority: int = 500) -> None:
        """添加一个 section。空内容自动跳过。"""
        stripped = content.strip()
        if not stripped:
            return
        self._sections.append(PromptSection(tag=tag, priority=priority, content=stripped))

    def sections(self) -> list[PromptSection]:
        """返回按优先级排序的 section 列表（只读副本）。"""
        return sorted(self._sections, key=lambda s: s.priority)

    def build(self) -> str:
        """组装为带 XML 标签的完整文本。"""
        sorted_sections = self.sections()
        if not sorted_sections:
            return ""
        parts: list[str] = []
        for section in sorted_sections:
            parts.append(f"<{section.tag}>\n{section.content}\n</{section.tag}>")
        return "\n\n".join(parts)

    def build_with_budget(self, max_tokens: int, chars_per_token: int = 4) -> str:
        """在 token 预算内组装。超出预算时按 COMPRESSIBLE_TAGS 顺序丢弃低优先级区块。

        保护标签内的 section 永远保留。可压缩标签按 COMPRESSIBLE_TAGS 顺序（最不重要的先丢弃）。
        """
        sorted_sections = self.sections()
        if not sorted_sections:
            return ""

        # 先尝试完整构建
        full_text = self._assemble(sorted_sections)
        if len(full_text) // chars_per_token <= max_tokens:
            return full_text

        # 超预算：逐步丢弃可压缩区块
        remaining = list(sorted_sections)
        for drop_tag in COMPRESSIBLE_TAGS:
            remaining = [s for s in remaining if s.tag != drop_tag]
            text = self._assemble(remaining)
            if len(text) // chars_per_token <= max_tokens:
                return text

        # 丢完所有可压缩区块仍超预算，返回剩余内容
        return self._assemble(remaining)

    @staticmethod
    def _assemble(sections: list[PromptSection]) -> str:
        if not sections:
            return ""
        parts = [f"<{s.tag}>\n{s.content}\n</{s.tag}>" for s in sections]
        return "\n\n".join(parts)
