"""浏览器控制 Agent：专注于 Web 自动化任务的预设代理。

与 default agent 的差异:
- system_prompt 专门介绍 browser_* 工具与 @ref 快照交互模型
- enable_browser 默认 True，且启动失败时直接报错（而非静默跳过）
- plugins/memory 默认启用但可关闭，保持专注时用户可精简
"""

from __future__ import annotations

from pathlib import Path

from ..llm.provider import LLMProvider
from ..types.tool import ToolSpec
from .base import Agent, AgentConfig

BROWSER_SYSTEM_PROMPT = (
    "You are topsport-browser-agent, specialized in web automation and information retrieval.\n\n"
    "## Browser toolset\n"
    "- `browser_navigate(url)`: open a URL; returns an accessibility snapshot with @ref labels\n"
    "- `browser_snapshot()`: refresh the snapshot of interactive elements\n"
    "- `browser_click(ref | selector)`: click element by @ref (preferred) or CSS selector\n"
    "- `browser_type(ref | selector, text)`: type into an input field\n"
    "- `browser_screenshot()`: capture current viewport to a file, returns the path\n"
    "- `browser_get_text(ref?)`: extract text from the page or a specific element\n\n"
    "## Interaction model\n"
    "Always call `browser_navigate` or `browser_snapshot` before interacting — "
    "you need fresh @refs (`@e1`, `@e2`, ...). Prefer @refs over CSS selectors: "
    "they are stable across re-snapshots and work even when the DOM lacks good selectors.\n\n"
    "## Workflow\n"
    "1. Navigate to the target URL\n"
    "2. Inspect the snapshot to locate the element you need\n"
    "3. Click / type using the @ref from the snapshot\n"
    "4. Re-snapshot after any navigation or state change\n"
    "5. Use `browser_get_text` to extract content; use `browser_screenshot` "
    "only when visual evidence is necessary (it's expensive).\n\n"
    "If a page fails to load or an element is missing, report the exact error — "
    "do not fabricate results."
)


class BrowserUnavailableError(RuntimeError):
    """Playwright 未安装或浏览器初始化失败。"""


def browser_agent(
    provider: LLMProvider,
    model: str,
    *,
    name: str = "browser",
    description: str = "Browser automation agent with Playwright-based web control",
    system_prompt: str | None = None,
    enable_memory: bool = True,
    enable_plugins: bool = True,
    enable_skills: bool = True,
    memory_base_path: Path | None = None,
    local_skill_dirs: list[Path] | None = None,
    extra_tools: list[ToolSpec] | None = None,
) -> Agent:
    """浏览器专精 Agent。启动后若没有注册到 browser_* 工具则抛 BrowserUnavailableError。"""
    config = AgentConfig(
        name=name,
        description=description,
        system_prompt=system_prompt or BROWSER_SYSTEM_PROMPT,
        model=model,
        enable_skills=enable_skills,
        enable_memory=enable_memory,
        enable_plugins=enable_plugins,
        enable_browser=True,
        memory_base_path=memory_base_path,
        local_skill_dirs=local_skill_dirs or [Path.home() / ".claude" / "skills"],
        extra_tools=list(extra_tools or []),
    )
    agent = Agent.from_config(provider, config)

    # 验证 browser 工具确实被注册 — 未装 playwright 时 Agent.from_config 会静默跳过，
    # 这里主动检测并抛错，让 browser agent 的契约清晰：没有 browser 就不该创建。
    if not _has_browser_tools(agent):
        raise BrowserUnavailableError(
            "browser_agent requires playwright. "
            "Install with: uv sync --group browser && playwright install chromium"
        )
    return agent


def _has_browser_tools(agent: Agent) -> bool:
    """检查 Engine 的 tool_sources 中是否有 browser 源。"""
    engine = agent.engine
    # Engine 没有公开 tool_sources 的访问器，走私有字段
    sources = getattr(engine, "_tool_sources", [])
    for source in sources:
        if getattr(source, "name", "") == "browser":
            return True
    return False
