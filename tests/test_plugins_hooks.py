"""Hook 执行器测试：解析、匹配、执行。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from topsport_agent.plugins.discovery import InstalledPlugin, discover_plugins
from topsport_agent.plugins.hook_runner import (
    PluginHook,
    PluginHookRunner,
    _expand_command,
    _match_target,
    _parse_hooks_json,
)
from topsport_agent.plugins.plugin import PluginDescriptor, scan_plugin
from topsport_agent.types.events import Event, EventType

# ---------------------------------------------------------------------------
# _parse_hooks_json 单元测试
# ---------------------------------------------------------------------------


def test_parse_hooks_basic(tmp_path: Path) -> None:
    """解析基本的 hooks.json。"""
    data = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "echo hello",
                            "async": False,
                            "timeout": 5,
                        }
                    ],
                }
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(path, tmp_path)
    assert len(result) == 1
    assert result[0].event == "SessionStart"
    assert result[0].matcher == "startup"
    assert result[0].command == "echo hello"
    assert result[0].is_async is False
    assert result[0].timeout == 5.0


def test_parse_hooks_multiple_events(tmp_path: Path) -> None:
    """多个事件各有 hooks。"""
    data = {
        "hooks": {
            "PreToolUse": [
                {"matcher": "Read|Write", "hooks": [{"type": "command", "command": "echo pre"}]}
            ],
            "PostToolUse": [
                {"hooks": [{"type": "command", "command": "echo post"}]}
            ],
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(path, tmp_path)
    assert len(result) == 2


def test_parse_hooks_skips_non_command(tmp_path: Path) -> None:
    """type 不是 command 的 hook 被跳过。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "script", "command": "echo skip"}]}
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(path, tmp_path)
    assert result == []


def test_parse_hooks_missing_file(tmp_path: Path) -> None:
    """文件不存在时返回空列表。"""
    result = _parse_hooks_json(tmp_path / "nonexistent.json", tmp_path)
    assert result == []


def test_parse_hooks_default_timeout(tmp_path: Path) -> None:
    """无 timeout 字段时默认 30 秒。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo hi"}]}
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(path, tmp_path)
    assert result[0].timeout == 30.0


def test_parse_hooks_default_async_true(tmp_path: Path) -> None:
    """无 async 字段时默认 True。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo hi"}]}
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(path, tmp_path)
    assert result[0].is_async is True


# ---------------------------------------------------------------------------
# _match_target / _expand_command 测试
# ---------------------------------------------------------------------------


def test_match_target_none_matcher() -> None:
    hook = PluginHook(
        event="X", matcher=None, command="", is_async=True, timeout=30, plugin_root=Path(".")
    )
    assert _match_target(hook, "anything") is True


def test_match_target_regex_match() -> None:
    hook = PluginHook(
        event="X", matcher="Read|Write", command="", is_async=True, timeout=30, plugin_root=Path(".")
    )
    assert _match_target(hook, "Read") is True
    assert _match_target(hook, "Write") is True
    assert _match_target(hook, "Bash") is False


def test_match_target_invalid_regex() -> None:
    hook = PluginHook(
        event="X", matcher="[invalid", command="", is_async=True, timeout=30, plugin_root=Path(".")
    )
    assert _match_target(hook, "test") is False


def test_expand_command() -> None:
    hook = PluginHook(
        event="X",
        matcher=None,
        command='"${CLAUDE_PLUGIN_ROOT}/hooks/run.sh" start',
        is_async=True,
        timeout=30,
        plugin_root=Path("/opt/plugin"),
    )
    assert _expand_command(hook) == '"/opt/plugin/hooks/run.sh" start'


# ---------------------------------------------------------------------------
# PluginHookRunner 测试
# ---------------------------------------------------------------------------


async def test_hook_runner_executes_sync_hook(tmp_path: Path) -> None:
    """同步 hook 执行外部命令。"""
    output_file = tmp_path / "output.txt"
    hook = PluginHook(
        event="SessionStart",
        matcher=None,
        command=f'{sys.executable} -c "from pathlib import Path; Path(\'{output_file}\').write_text(\'ok\')"',
        is_async=False,
        timeout=30,
        plugin_root=tmp_path,
    )
    runner = PluginHookRunner([hook])
    event = Event(type=EventType.RUN_START, session_id="s1")
    await runner.on_event(event)
    assert output_file.read_text() == "ok"


async def test_hook_runner_skips_unmatched_event() -> None:
    """事件类型不匹配时不执行。"""
    hook = PluginHook(
        event="SessionEnd",
        matcher=None,
        command="echo should-not-run",
        is_async=False,
        timeout=30,
        plugin_root=Path("."),
    )
    runner = PluginHookRunner([hook])
    # RUN_START -> SessionStart，不匹配 SessionEnd
    event = Event(type=EventType.RUN_START, session_id="s1")
    await runner.on_event(event)
    # 无异常即通过


async def test_hook_runner_tool_matcher(tmp_path: Path) -> None:
    """PreToolUse hook 的 matcher 匹配工具名。"""
    output_file = tmp_path / "tool_output.txt"
    hook = PluginHook(
        event="PreToolUse",
        matcher="Read|Write",
        command=f'{sys.executable} -c "from pathlib import Path; Path(\'{output_file}\').write_text(\'matched\')"',
        is_async=False,
        timeout=30,
        plugin_root=tmp_path,
    )
    runner = PluginHookRunner([hook])

    # 匹配的工具名
    event = Event(
        type=EventType.TOOL_CALL_START,
        session_id="s1",
        payload={"name": "Read", "call_id": "c1"},
    )
    await runner.on_event(event)
    assert output_file.read_text() == "matched"

    # 不匹配的工具名
    output_file.unlink()
    event2 = Event(
        type=EventType.TOOL_CALL_START,
        session_id="s1",
        payload={"name": "Bash", "call_id": "c2"},
    )
    await runner.on_event(event2)
    assert not output_file.exists()


async def test_hook_runner_timeout_no_crash(tmp_path: Path) -> None:
    """hook 超时时不抛异常。"""
    hook = PluginHook(
        event="SessionStart",
        matcher=None,
        command=f"{sys.executable} -c \"import time; time.sleep(10)\"",
        is_async=False,
        timeout=0.5,
        plugin_root=tmp_path,
    )
    runner = PluginHookRunner([hook])
    event = Event(type=EventType.RUN_START, session_id="s1")
    # 不应抛异常
    await runner.on_event(event)


async def test_hook_runner_command_failure_no_crash() -> None:
    """命令失败时不抛异常。"""
    hook = PluginHook(
        event="SessionStart",
        matcher=None,
        command="nonexistent-command-xyz",
        is_async=False,
        timeout=5,
        plugin_root=Path("."),
    )
    runner = PluginHookRunner([hook])
    event = Event(type=EventType.RUN_START, session_id="s1")
    await runner.on_event(event)


async def test_hook_runner_ignores_irrelevant_events() -> None:
    """STEP_START 等不在映射中的事件被忽略。"""
    hook = PluginHook(
        event="SessionStart",
        matcher=None,
        command="echo hello",
        is_async=False,
        timeout=5,
        plugin_root=Path("."),
    )
    runner = PluginHookRunner([hook])
    event = Event(type=EventType.STEP_START, session_id="s1", payload={"step": 0})
    await runner.on_event(event)


def test_from_plugins(tmp_path: Path) -> None:
    """from_plugins 合并多个插件的 hooks。"""
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir()
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo a"}]}
            ]
        }
    }
    (hooks_dir / "hooks.json").write_text(json.dumps(data))

    desc = PluginDescriptor(
        info=InstalledPlugin(name="p", marketplace="m", install_path=tmp_path, version="1.0"),
        hooks_config=hooks_dir / "hooks.json",
    )
    runner = PluginHookRunner.from_plugins([desc])
    assert len(runner._hooks) == 1


# ---------------------------------------------------------------------------
# 集成测试：真实插件
# ---------------------------------------------------------------------------

_REAL_PLUGINS_DIR = Path.home() / ".claude" / "plugins"


@pytest.mark.skipif(
    not (_REAL_PLUGINS_DIR / "installed_plugins.json").is_file(),
    reason="~/.claude/plugins/installed_plugins.json not found",
)
class TestRealHooks:
    def test_parse_real_hooks_no_crash(self) -> None:
        """解析所有真实插件的 hooks.json 不抛异常。"""
        plugins = discover_plugins(_REAL_PLUGINS_DIR)
        descs = [scan_plugin(p) for p in plugins]
        runner = PluginHookRunner.from_plugins(descs)
        assert isinstance(runner._hooks, list)

    def test_superpowers_has_hooks(self) -> None:
        """superpowers 插件应有至少一条 hook。"""
        plugins = discover_plugins(_REAL_PLUGINS_DIR)
        sp = next((p for p in plugins if p.name == "superpowers"), None)
        assert sp is not None
        desc = scan_plugin(sp)
        runner = PluginHookRunner.from_plugins([desc])
        assert len(runner._hooks) > 0
