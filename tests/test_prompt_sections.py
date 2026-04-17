"""Prompt Section 机制测试：PromptBuilder + 标签组装 + 预算裁剪 + 压缩集成。"""

from __future__ import annotations

from topsport_agent.engine.compaction.auto import (
    compact_system_prompt,
    extract_protected_sections,
)
from topsport_agent.engine.prompt import (
    COMPRESSIBLE_TAGS,
    PROTECTED_TAGS,
    PromptBuilder,
    SectionPriority,
)

# ---------------------------------------------------------------------------
# PromptBuilder 基础测试
# ---------------------------------------------------------------------------


def test_builder_empty() -> None:
    """空 builder 返回空字符串。"""
    builder = PromptBuilder()
    assert builder.build() == ""


def test_builder_single_section() -> None:
    """单个 section 带 XML 标签。"""
    builder = PromptBuilder()
    builder.add("system-prompt", "You are helpful.", 0)
    result = builder.build()
    assert "<system-prompt>" in result
    assert "You are helpful." in result
    assert "</system-prompt>" in result


def test_builder_skips_empty_content() -> None:
    """空内容自动跳过。"""
    builder = PromptBuilder()
    builder.add("empty", "", 0)
    builder.add("whitespace", "   \n  ", 0)
    builder.add("real", "content", 100)
    assert len(builder.sections()) == 1


def test_builder_priority_ordering() -> None:
    """按 priority 升序排列。"""
    builder = PromptBuilder()
    builder.add("low", "low priority", 900)
    builder.add("high", "high priority", 0)
    builder.add("mid", "mid priority", 500)
    sections = builder.sections()
    assert sections[0].tag == "high"
    assert sections[1].tag == "mid"
    assert sections[2].tag == "low"


def test_builder_xml_structure() -> None:
    """完整 XML 标签结构。"""
    builder = PromptBuilder()
    builder.add("system-prompt", "You are helpful.", SectionPriority.SYSTEM_PROMPT)
    builder.add("working-memory", "goal: refactor", SectionPriority.WORKING_MEMORY)
    result = builder.build()
    # system-prompt 在前
    sp_pos = result.index("<system-prompt>")
    wm_pos = result.index("<working-memory>")
    assert sp_pos < wm_pos
    # 各区块都有闭合标签
    assert "</system-prompt>" in result
    assert "</working-memory>" in result


def test_builder_multiple_same_priority() -> None:
    """同优先级的 section 保持添加顺序。"""
    builder = PromptBuilder()
    builder.add("a", "first", 500)
    builder.add("b", "second", 500)
    sections = builder.sections()
    assert sections[0].tag == "a"
    assert sections[1].tag == "b"


# ---------------------------------------------------------------------------
# build_with_budget 测试
# ---------------------------------------------------------------------------


def test_budget_within_limit() -> None:
    """预算充足时返回完整内容。"""
    builder = PromptBuilder()
    builder.add("system-prompt", "short", 0)
    result = builder.build_with_budget(max_tokens=1000)
    assert "<system-prompt>" in result
    assert "short" in result


def test_budget_drops_compressible() -> None:
    """超出预算时按 COMPRESSIBLE_TAGS 顺序丢弃。"""
    builder = PromptBuilder()
    builder.add("system-prompt", "core", 0)
    builder.add("skills-catalog", "x" * 4000, 300)
    builder.add("working-memory", "important memo", 200)

    # 给很少的预算
    result = builder.build_with_budget(max_tokens=50, chars_per_token=4)
    # system-prompt 受保护，应该保留
    assert "<system-prompt>" in result
    # 可压缩的应被丢弃
    assert "x" * 4000 not in result


def test_budget_preserves_protected() -> None:
    """即使预算极低，protected 标签的内容也不丢弃。"""
    builder = PromptBuilder()
    builder.add("system-prompt", "You are helpful.", 0)
    builder.add("instructions", "Must follow rules.", 900)
    result = builder.build_with_budget(max_tokens=5, chars_per_token=4)
    assert "<system-prompt>" in result
    assert "<instructions>" in result


# ---------------------------------------------------------------------------
# compact_system_prompt 测试
# ---------------------------------------------------------------------------


def test_compact_drop_specific_tags() -> None:
    """按标签名丢弃指定 section。"""
    text = (
        "<system-prompt>\ncore\n</system-prompt>\n\n"
        "<skills-catalog>\nskill list\n</skills-catalog>\n\n"
        "<working-memory>\nmemory\n</working-memory>"
    )
    result = compact_system_prompt(text, drop_tags=frozenset({"skills-catalog"}))
    assert "<system-prompt>" in result
    assert "<working-memory>" in result
    assert "<skills-catalog>" not in result


def test_compact_protects_system_prompt() -> None:
    """即使要求丢弃 system-prompt，也会被保护。"""
    text = "<system-prompt>\ncore\n</system-prompt>"
    result = compact_system_prompt(text, drop_tags=frozenset({"system-prompt"}))
    assert "<system-prompt>" in result


def test_compact_none_drop_tags() -> None:
    """drop_tags=None 时原样返回。"""
    text = "<a>\ncontent\n</a>"
    result = compact_system_prompt(text, drop_tags=None)
    assert result == text


# ---------------------------------------------------------------------------
# extract_protected_sections 测试
# ---------------------------------------------------------------------------


def test_extract_protected_only() -> None:
    """只保留 protected 标签。"""
    text = (
        "<system-prompt>\ncore\n</system-prompt>\n\n"
        "<skills-catalog>\nskills\n</skills-catalog>\n\n"
        "<identity>\nI am X\n</identity>\n\n"
        "<working-memory>\nmemory\n</working-memory>"
    )
    result = extract_protected_sections(text)
    assert "<system-prompt>" in result
    assert "<identity>" in result
    assert "<skills-catalog>" not in result
    assert "<working-memory>" not in result


def test_extract_no_protected_returns_original() -> None:
    """无 protected 标签时返回原始文本。"""
    text = "<custom>\ncontent\n</custom>"
    result = extract_protected_sections(text)
    assert result == text


# ---------------------------------------------------------------------------
# SectionPriority 常量测试
# ---------------------------------------------------------------------------


def test_section_priority_ordering() -> None:
    """优先级值从小到大。"""
    assert SectionPriority.SYSTEM_PROMPT < SectionPriority.IDENTITY
    assert SectionPriority.IDENTITY < SectionPriority.WORKING_MEMORY
    assert SectionPriority.WORKING_MEMORY < SectionPriority.SKILLS_CATALOG
    assert SectionPriority.SKILLS_CATALOG < SectionPriority.ACTIVE_SKILLS
    assert SectionPriority.ACTIVE_SKILLS < SectionPriority.PLUGIN_CONTEXT
    assert SectionPriority.PLUGIN_CONTEXT < SectionPriority.TOOLS_GUIDE
    assert SectionPriority.TOOLS_GUIDE < SectionPriority.SESSION_STATE
    assert SectionPriority.SESSION_STATE < SectionPriority.INSTRUCTIONS


# ---------------------------------------------------------------------------
# 集成测试：injector 输出带 section 元信息
# ---------------------------------------------------------------------------


def test_prompt_constants() -> None:
    """PROTECTED_TAGS 和 COMPRESSIBLE_TAGS 不重叠。"""
    overlap = PROTECTED_TAGS & set(COMPRESSIBLE_TAGS)
    assert overlap == set(), f"protected and compressible overlap: {overlap}"
