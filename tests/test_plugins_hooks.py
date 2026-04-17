"""Hook 执行器测试：解析、匹配、执行。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from topsport_agent.plugins import (
    PluginPolicyViolation,
    PluginSecurityPolicy,
)
from topsport_agent.plugins.discovery import InstalledPlugin, discover_plugins
from topsport_agent.plugins.hook_runner import (
    PluginHook,
    PluginHookRunner,
    _expand_argv,
    _match_target,
    _parse_hooks_json,
)
from topsport_agent.plugins.plugin import PluginDescriptor, scan_plugin
from topsport_agent.types.events import Event, EventType

_PERMISSIVE = PluginSecurityPolicy.permissive()


# ---------------------------------------------------------------------------
# _parse_hooks_json 单元测试
# ---------------------------------------------------------------------------


def test_parse_hooks_basic_argv(tmp_path: Path) -> None:
    """argv 列表形式 command，走 exec 路径。"""
    data = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup",
                    "hooks": [
                        {
                            "type": "command",
                            "command": ["echo", "hello"],
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

    result = _parse_hooks_json(
        path, tmp_path, plugin_name="p", policy=_PERMISSIVE
    )
    assert len(result) == 1
    assert result[0].event == "SessionStart"
    assert result[0].matcher == "startup"
    assert result[0].argv == ["echo", "hello"]
    assert result[0].is_async is False
    assert result[0].timeout == 5.0


def test_parse_hooks_legacy_string_permissive(
    tmp_path: Path, caplog
) -> None:
    """permissive 下 legacy 字符串 command 被 shlex 切分，记 warning。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo hi there"}]}
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    with caplog.at_level("WARNING", logger="topsport_agent.plugins.policy"):
        result = _parse_hooks_json(
            path, tmp_path, plugin_name="legacy-plugin", policy=_PERMISSIVE
        )
    assert len(result) == 1
    assert result[0].argv == ["echo", "hi", "there"]
    assert any(
        "legacy string command" in rec.message for rec in caplog.records
    )


def test_parse_hooks_legacy_string_strict_rejected(
    tmp_path: Path, caplog
) -> None:
    """strict 下 legacy 字符串 command 被拒绝，该条目跳过不抛整体异常。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo hi"}]},
                {"hooks": [{"type": "command", "command": ["echo", "ok"]}]},
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    policy = PluginSecurityPolicy.strict(trusted=["p"])
    with caplog.at_level("WARNING"):
        result = _parse_hooks_json(
            path, tmp_path, plugin_name="p", policy=policy
        )
    # 只有 argv 形式的那条活下来
    assert len(result) == 1
    assert result[0].argv == ["echo", "ok"]


def test_parse_hooks_multiple_events(tmp_path: Path) -> None:
    """多个事件各有 hooks。"""
    data = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Read|Write",
                    "hooks": [{"type": "command", "command": ["echo", "pre"]}],
                }
            ],
            "PostToolUse": [
                {"hooks": [{"type": "command", "command": ["echo", "post"]}]}
            ],
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(
        path, tmp_path, plugin_name="p", policy=_PERMISSIVE
    )
    assert len(result) == 2


def test_parse_hooks_skips_non_command(tmp_path: Path) -> None:
    """type 不是 command 的 hook 被跳过。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "script", "command": ["echo", "skip"]}]}
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(
        path, tmp_path, plugin_name="p", policy=_PERMISSIVE
    )
    assert result == []


def test_parse_hooks_missing_file(tmp_path: Path) -> None:
    """文件不存在时返回空列表。"""
    result = _parse_hooks_json(
        tmp_path / "nonexistent.json",
        tmp_path,
        plugin_name="p",
        policy=_PERMISSIVE,
    )
    assert result == []


def test_parse_hooks_default_timeout(tmp_path: Path) -> None:
    """无 timeout 字段时默认 30 秒。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": ["echo", "hi"]}]}
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(
        path, tmp_path, plugin_name="p", policy=_PERMISSIVE
    )
    assert result[0].timeout == 30.0


def test_parse_hooks_default_async_true(tmp_path: Path) -> None:
    """无 async 字段时默认 True。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": ["echo", "hi"]}]}
            ]
        }
    }
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps(data))

    result = _parse_hooks_json(
        path, tmp_path, plugin_name="p", policy=_PERMISSIVE
    )
    assert result[0].is_async is True


# ---------------------------------------------------------------------------
# _match_target / _expand_argv 测试
# ---------------------------------------------------------------------------


def test_match_target_none_matcher() -> None:
    hook = PluginHook(
        event="X", matcher=None, argv=[], is_async=True, timeout=30, plugin_root=Path(".")
    )
    assert _match_target(hook, "anything") is True


def test_match_target_regex_match() -> None:
    hook = PluginHook(
        event="X", matcher="Read|Write", argv=[], is_async=True, timeout=30, plugin_root=Path(".")
    )
    assert _match_target(hook, "Read") is True
    assert _match_target(hook, "Write") is True
    assert _match_target(hook, "Bash") is False


def test_match_target_invalid_regex() -> None:
    hook = PluginHook(
        event="X", matcher="[invalid", argv=[], is_async=True, timeout=30, plugin_root=Path(".")
    )
    assert _match_target(hook, "test") is False


def test_expand_argv_replaces_plugin_root() -> None:
    hook = PluginHook(
        event="X",
        matcher=None,
        argv=["${CLAUDE_PLUGIN_ROOT}/hooks/run.sh", "start"],
        is_async=True,
        timeout=30,
        plugin_root=Path("/opt/plugin"),
    )
    assert _expand_argv(hook) == ["/opt/plugin/hooks/run.sh", "start"]


def test_expand_argv_does_not_interpolate_tool_name() -> None:
    """关键安全断言：TOOL_NAME 不能出现在 argv 扩展里（只进 env）。"""
    hook = PluginHook(
        event="PreToolUse",
        matcher=None,
        argv=["${TOOL_NAME}"],  # 即便用户写成这样也不扩展
        is_async=True,
        timeout=30,
        plugin_root=Path("/opt"),
    )
    # 字面保留，不做 TOOL_NAME 替换
    assert _expand_argv(hook) == ["${TOOL_NAME}"]


# ---------------------------------------------------------------------------
# PluginHookRunner 测试
# ---------------------------------------------------------------------------


async def test_hook_runner_executes_sync_hook(tmp_path: Path) -> None:
    """同步 hook 执行外部命令（argv 形式）。"""
    output_file = tmp_path / "output.txt"
    hook = PluginHook(
        event="SessionStart",
        matcher=None,
        argv=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path(r'{output_file}').write_text('ok')",
        ],
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
        argv=["echo", "should-not-run"],
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
        argv=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path(r'{output_file}').write_text('matched')",
        ],
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
        argv=[sys.executable, "-c", "import time; time.sleep(10)"],
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
        argv=["nonexistent-command-xyz"],
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
        argv=["echo", "hello"],
        is_async=False,
        timeout=5,
        plugin_root=Path("."),
    )
    runner = PluginHookRunner([hook])
    event = Event(type=EventType.STEP_START, session_id="s1", payload={"step": 0})
    await runner.on_event(event)


async def test_hook_runner_does_not_execute_shell_syntax(tmp_path: Path) -> None:
    """关键安全断言：argv 中的 shell 元字符不会被解释为 shell 语法。

    如果命令被 shell 执行，`touch pwned` 会产生文件。走 exec 路径时，
    `;`/`touch pwned` 被当作字面参数传给第一个程序，不会创建文件。
    """
    pwned_marker = tmp_path / "pwned"
    hook = PluginHook(
        event="SessionStart",
        matcher=None,
        argv=["echo", "ok;", "touch", str(pwned_marker)],
        is_async=False,
        timeout=5,
        plugin_root=tmp_path,
    )
    runner = PluginHookRunner([hook])
    event = Event(type=EventType.RUN_START, session_id="s1")
    await runner.on_event(event)
    # echo 把 ";" "touch" path 当普通参数打印，不会真执行 touch
    assert not pwned_marker.exists()


# ---------------------------------------------------------------------------
# from_plugins + 策略集成
# ---------------------------------------------------------------------------


def _make_desc(tmp_path: Path, name: str, data: dict) -> PluginDescriptor:
    hooks_dir = tmp_path / name / "hooks"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "hooks.json").write_text(json.dumps(data))
    return PluginDescriptor(
        info=InstalledPlugin(
            name=name,
            marketplace="m",
            install_path=tmp_path / name,
            version="1.0",
        ),
        hooks_config=hooks_dir / "hooks.json",
    )


def test_from_plugins_permissive(tmp_path: Path) -> None:
    """permissive 下所有插件的 hooks 都被加载。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": ["echo", "a"]}]}
            ]
        }
    }
    desc = _make_desc(tmp_path, "p1", data)
    runner = PluginHookRunner.from_plugins([desc])
    assert len(runner._hooks) == 1


def test_from_plugins_strict_only_loads_trusted(tmp_path: Path) -> None:
    """strict 下非 trusted_plugins 的插件 hooks 被拒绝。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": ["echo", "a"]}]}
            ]
        }
    }
    trusted = _make_desc(tmp_path, "good", data)
    untrusted = _make_desc(tmp_path, "evil", data)

    policy = PluginSecurityPolicy.strict(trusted=["good"])
    runner = PluginHookRunner.from_plugins(
        [trusted, untrusted], policy=policy
    )
    # good 的一条被加载，evil 的整体跳过
    assert len(runner._hooks) == 1


def test_from_plugins_strict_rejects_legacy_string_command(
    tmp_path: Path,
) -> None:
    """strict + trusted，但 command 是 legacy 字符串 —— 该条目仍被拒绝。"""
    data = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo hi"}]}
            ]
        }
    }
    desc = _make_desc(tmp_path, "p", data)
    policy = PluginSecurityPolicy.strict(trusted=["p"])
    runner = PluginHookRunner.from_plugins([desc], policy=policy)
    assert runner._hooks == []


def test_enforce_command_shape_rejects_non_string_non_list() -> None:
    """其它类型 command（比如 int）直接抛 PluginPolicyViolation。"""
    from topsport_agent.plugins.policy import enforce_command_shape

    with pytest.raises(PluginPolicyViolation, match="must be a list"):
        enforce_command_shape(
            plugin_name="p",
            event_name="SessionStart",
            command=42,
            policy=_PERMISSIVE,
        )


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
        """解析所有真实插件的 hooks.json 不抛异常（permissive 兼容）。"""
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
