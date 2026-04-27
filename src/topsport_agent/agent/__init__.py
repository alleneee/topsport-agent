"""Agent 抽象层：封装 Engine + 扩展能力为高层对象。

导出:
- Agent / AgentConfig: 基础类
- default_agent(): 通用默认 Agent 工厂
- browser_agent(): 浏览器自动化专精 Agent 工厂
- extract_assistant_text(): 从事件流中抽取最终回复
"""

from .base import Agent, AgentConfig, extract_assistant_text
from .browser import BROWSER_SYSTEM_PROMPT, BrowserUnavailableError, browser_agent
from .config_parts import AgentIdentity, CapabilityRegistry, CapabilityToggles
from .default import DEFAULT_SYSTEM_PROMPT, default_agent

__all__ = [
    "BROWSER_SYSTEM_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "Agent",
    "AgentConfig",
    "AgentIdentity",
    "BrowserUnavailableError",
    "CapabilityRegistry",
    "CapabilityToggles",
    "browser_agent",
    "default_agent",
    "extract_assistant_text",
]
