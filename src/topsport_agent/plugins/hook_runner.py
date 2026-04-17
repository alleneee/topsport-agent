"""Hook 执行器：解析 hooks.json，桥接 Engine 事件到外部命令。

Claude plugin hooks 在特定生命周期事件触发时运行外部 shell 命令。
PluginHookRunner 作为 EventSubscriber 注入 Engine，将引擎事件映射为
Claude hook 事件名，正则匹配后通过子进程执行。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..types.events import Event, EventType
from .plugin import PluginDescriptor

_logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0

# Engine EventType -> Claude hook 事件名的映射
_EVENT_MAP: dict[EventType, str] = {
    EventType.RUN_START: "SessionStart",
    EventType.RUN_END: "SessionEnd",
    EventType.TOOL_CALL_START: "PreToolUse",
    EventType.TOOL_CALL_END: "PostToolUse",
}


@dataclass(slots=True)
class PluginHook:
    """一条 hook 规则。"""

    event: str  # Claude hook 事件名
    matcher: str | None  # 正则模式，None 表示全部匹配
    command: str  # shell 命令模板
    is_async: bool  # True 时 fire-and-forget
    timeout: float  # 秒
    plugin_root: Path  # ${CLAUDE_PLUGIN_ROOT} 展开用


def _parse_hooks_json(path: Path, plugin_root: Path) -> list[PluginHook]:
    """解析一个 hooks.json 文件为 PluginHook 列表。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning("failed to read hooks.json at %s: %s", path, exc)
        return []

    hooks_map = data.get("hooks", {})
    result: list[PluginHook] = []

    for event_name, entries in hooks_map.items():
        for entry in entries:
            matcher = entry.get("matcher")
            for hook_def in entry.get("hooks", []):
                if hook_def.get("type") != "command":
                    continue
                command = hook_def.get("command", "")
                if not command:
                    continue
                is_async = hook_def.get("async", True)
                timeout = float(hook_def.get("timeout", _DEFAULT_TIMEOUT))
                result.append(
                    PluginHook(
                        event=event_name,
                        matcher=matcher,
                        command=command,
                        is_async=is_async,
                        timeout=timeout,
                        plugin_root=plugin_root,
                    )
                )

    return result


def _match_target(hook: PluginHook, target: str | None) -> bool:
    """检查 hook 的 matcher 正则是否匹配目标字符串。"""
    if hook.matcher is None:
        return True
    if target is None:
        return True
    try:
        return re.search(hook.matcher, target) is not None
    except re.error:
        _logger.warning("invalid hook matcher regex: %s", hook.matcher)
        return False


def _expand_command(hook: PluginHook) -> str:
    """展开命令模板中的 ${CLAUDE_PLUGIN_ROOT}。"""
    return hook.command.replace("${CLAUDE_PLUGIN_ROOT}", str(hook.plugin_root))


def _match_target_for_event(event: Event, claude_event: str) -> str | None:
    """根据 Claude hook 事件类型，从 Engine event payload 中提取匹配目标。"""
    if claude_event in ("PreToolUse", "PostToolUse"):
        return event.payload.get("name")
    return None


class PluginHookRunner:
    """EventSubscriber 实现：接收 Engine 事件，匹配后执行外部命令。

    命令执行失败只 log warning，不中断引擎。
    """

    name = "plugin_hooks"

    def __init__(self, hooks: list[PluginHook] | None = None) -> None:
        self._hooks = hooks or []
        self._fire_and_forget_tasks: set[asyncio.Task[None]] = set()

    async def on_event(self, event: Event) -> None:
        claude_event = _EVENT_MAP.get(event.type)
        if claude_event is None:
            # MESSAGE_APPENDED with role=user -> UserPromptSubmit
            if (
                event.type == EventType.MESSAGE_APPENDED
                and event.payload.get("role") == "user"
            ):
                claude_event = "UserPromptSubmit"
            else:
                return

        target = _match_target_for_event(event, claude_event)

        for hook in self._hooks:
            if hook.event != claude_event:
                continue
            if not _match_target(hook, target):
                continue

            if hook.is_async:
                task = asyncio.create_task(self._execute(hook, event))
                self._fire_and_forget_tasks.add(task)
                task.add_done_callback(self._fire_and_forget_tasks.discard)
            else:
                await self._execute(hook, event)

    async def _execute(self, hook: PluginHook, event: Event) -> None:
        """执行单条 hook 命令。"""
        command = _expand_command(hook)
        env = os.environ.copy()
        env["CLAUDE_PLUGIN_ROOT"] = str(hook.plugin_root)
        env["SESSION_ID"] = event.session_id
        # 工具相关事件注入 TOOL_NAME
        tool_name = event.payload.get("name", "")
        if tool_name:
            env["TOOL_NAME"] = tool_name

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            await asyncio.wait_for(proc.communicate(), timeout=hook.timeout)
        except TimeoutError:
            _logger.warning(
                "hook timed out (%.0fs): %s [event=%s]",
                hook.timeout,
                command[:80],
                hook.event,
            )
        except Exception as exc:
            _logger.warning(
                "hook execution failed: %s [event=%s, error=%s]",
                command[:80],
                hook.event,
                exc,
            )

    @classmethod
    def from_plugins(cls, plugins: list[PluginDescriptor]) -> PluginHookRunner:
        """合并所有插件的 hooks.json 为统一的 hook 列表。"""
        all_hooks: list[PluginHook] = []
        for plugin in plugins:
            if plugin.hooks_config is not None:
                hooks = _parse_hooks_json(
                    plugin.hooks_config, plugin.info.install_path
                )
                all_hooks.extend(hooks)
        return cls(all_hooks)
