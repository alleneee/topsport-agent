"""OpenSandboxPool + OpenSandboxToolSource 的单元测试。

不依赖 opensandbox 包：通过 sandbox_factory 注入 MockSandbox，通过
write_entry_cls 注入 MockWriteEntry。整套测试在未安装 opensandbox 时依然全绿。
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from topsport_agent.sandbox import (
    OpenSandboxPool,
    OpenSandboxToolSource,
    SessionSandboxBinding,
)
from topsport_agent.sandbox.pool import TenantQuotaExceeded
from topsport_agent.types.tool import ToolContext


class MockExecution:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class MockCommands:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, command: str) -> MockExecution:
        self.calls.append(command)
        if command == "fail":
            return MockExecution(stderr="boom", exit_code=1)
        if command == "raise":
            raise RuntimeError("transport down")
        return MockExecution(stdout=f"out:{command}")


class MockFiles:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.reads: list[str] = []
        self.writes: list[tuple[str, str]] = []

    async def read_file(self, path: str) -> str:
        self.reads.append(path)
        if path not in self._store:
            raise FileNotFoundError(path)
        return self._store[path]

    async def write_files(self, entries: list) -> None:
        for entry in entries:
            self._store[entry.path] = entry.data
            self.writes.append((entry.path, entry.data))


class MockEndpoint:
    """与真实 SandboxEndpoint 最小兼容：endpoint 字符串 + headers dict。"""
    def __init__(self, endpoint: str = "fake-host") -> None:
        self.endpoint = endpoint
        self.headers: dict[str, str] = {}


class MockSandbox:
    instance_counter = 0

    def __init__(self, session_id: str) -> None:
        MockSandbox.instance_counter += 1
        self.session_id = session_id
        self.id = f"sb-{session_id}-{MockSandbox.instance_counter}"
        self.commands = MockCommands()
        self.files = MockFiles()
        self.killed = False
        self.paused = False

    async def kill(self) -> None:
        self.killed = True

    async def pause(self) -> None:
        self.paused = True

    async def get_endpoint(self, _port: int) -> MockEndpoint:
        # 真实 Sandbox.get_endpoint 是 async；fast_exec 对它 await。
        return MockEndpoint(endpoint=f"sb-{self.session_id}.local")


class MockWriteEntry:
    def __init__(self, *, path: str, data: str) -> None:
        self.path = path
        self.data = data


def make_pool() -> OpenSandboxPool:
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    return OpenSandboxPool(sandbox_factory=factory)


def _sse_body(*events: dict) -> bytes:
    """把 JSON 事件列表拼成 execd /command 端点的 SSE 响应体（裸 JSON 行）。"""
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


def make_fast_pool(handler) -> OpenSandboxPool:
    """构造带 httpx MockTransport 的 pool，供 fast_shell 单测使用。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    def http_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return OpenSandboxPool(sandbox_factory=factory, http_client_factory=http_factory)


def _ctx(session_id: str = "s1", call_id: str = "c1") -> ToolContext:
    return ToolContext(
        session_id=session_id, call_id=call_id, cancel_event=asyncio.Event()
    )


# ---------- pool lifecycle ----------

async def test_pool_acquire_returns_same_sandbox_for_session() -> None:
    pool = make_pool()
    sb1 = await pool.acquire("s1")
    sb2 = await pool.acquire("s1")
    assert sb1 is sb2


async def test_pool_acquire_isolates_sessions() -> None:
    pool = make_pool()
    sb1 = await pool.acquire("s1")
    sb2 = await pool.acquire("s2")
    assert sb1 is not sb2
    assert sb1.session_id == "s1"
    assert sb2.session_id == "s2"


async def test_pool_concurrent_acquire_creates_once() -> None:
    """10 个协程同时 acquire 同一 session，factory 只应被调用一次。"""
    created = 0

    async def slow_factory(session_id: str) -> MockSandbox:
        nonlocal created
        await asyncio.sleep(0.01)
        created += 1
        return MockSandbox(session_id)

    pool = OpenSandboxPool(sandbox_factory=slow_factory)
    sandboxes = await asyncio.gather(*(pool.acquire("s1") for _ in range(10)))
    assert created == 1
    assert all(sb is sandboxes[0] for sb in sandboxes)


async def test_pool_release_kills_and_removes() -> None:
    pool = make_pool()
    sb = await pool.acquire("s1")
    await pool.release("s1")
    assert sb.killed is True
    assert not pool.has("s1")


async def test_pool_release_is_idempotent() -> None:
    pool = make_pool()
    await pool.acquire("s1")
    await pool.release("s1")
    await pool.release("s1")
    await pool.release("never-existed")


async def test_pool_release_tolerates_kill_failure() -> None:
    """kill 抛异常时 pool 仍应移除条目，不让状态残留。"""

    class BrokenSandbox(MockSandbox):
        async def kill(self) -> None:
            raise RuntimeError("kill failed")

    async def factory(sid: str) -> BrokenSandbox:
        return BrokenSandbox(sid)

    pool = OpenSandboxPool(sandbox_factory=factory)
    await pool.acquire("s1")
    await pool.release("s1")
    assert not pool.has("s1")


async def test_pool_release_by_prefix_matches_and_releases() -> None:
    pool = make_pool()
    sb_a1 = await pool.acquire("plan-1:step-a")
    sb_a2 = await pool.acquire("plan-1:step-b")
    sb_other = await pool.acquire("plan-2:step-x")
    sb_chat = await pool.acquire("chat-sid")

    released = await pool.release_by_prefix("plan-1:")
    assert released == 2
    assert sb_a1.killed and sb_a2.killed
    assert not sb_other.killed and not sb_chat.killed
    assert not pool.has("plan-1:step-a")
    assert not pool.has("plan-1:step-b")
    assert pool.has("plan-2:step-x")
    assert pool.has("chat-sid")
    await pool.close_all()


async def test_pool_release_by_prefix_empty_prefix_noop() -> None:
    """空 prefix 不做任何事（防守：不想因传 '' 而误清全部）。"""
    pool = make_pool()
    await pool.acquire("a")
    released = await pool.release_by_prefix("")
    assert released == 0
    assert pool.has("a")
    await pool.close_all()


async def test_pool_release_by_prefix_no_match_returns_zero() -> None:
    pool = make_pool()
    await pool.acquire("plan-1:step-a")
    released = await pool.release_by_prefix("plan-999:")
    assert released == 0
    assert pool.has("plan-1:step-a")
    await pool.close_all()


async def test_pool_close_all_clears_every_session() -> None:
    pool = make_pool()
    sb1 = await pool.acquire("s1")
    sb2 = await pool.acquire("s2")
    await pool.close_all()
    assert sb1.killed is True
    assert sb2.killed is True
    assert not pool.has("s1")
    assert not pool.has("s2")


async def test_pool_factory_failure_not_cached() -> None:
    """factory 首次失败不缓存，下次 acquire 重试。"""
    call_count = 0

    async def flaky_factory(session_id: str) -> MockSandbox:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        return MockSandbox(session_id)

    pool = OpenSandboxPool(sandbox_factory=flaky_factory)
    with pytest.raises(RuntimeError):
        await pool.acquire("s1")
    sb = await pool.acquire("s1")
    assert sb is not None
    assert pool.has("s1")


async def test_pool_release_after_acquire_allows_new_sandbox() -> None:
    pool = make_pool()
    sb_first = await pool.acquire("s1")
    await pool.release("s1")
    sb_second = await pool.acquire("s1")
    assert sb_first is not sb_second


# ---------- idle pause / resume ----------

async def test_pool_idle_pause_disabled_by_default() -> None:
    """构造未传 idle_pause_seconds → reaper 不应启动，sandbox 不会被 pause。"""
    pool = make_pool()
    sb = await pool.acquire("s1")
    assert pool._reaper_task is None
    # 稍等，确保没有后台任务悄悄触发 pause
    await asyncio.sleep(0.05)
    assert not sb.paused
    assert not pool.is_paused("s1")
    await pool.close_all()


async def test_pool_idle_pause_triggers_after_threshold() -> None:
    """空闲超过 idle_pause_seconds 后 reaper 调 sandbox.pause()。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        idle_pause_seconds=0.05,
        reaper_interval_seconds=0.02,
    )
    sb = await pool.acquire("s1")
    # 等到空闲超阈值 + reaper 至少扫过一轮
    for _ in range(50):
        await asyncio.sleep(0.02)
        if pool.is_paused("s1"):
            break
    assert sb.paused is True
    assert pool.is_paused("s1")
    await pool.close_all()


async def test_pool_acquire_resumes_paused_sandbox() -> None:
    """被 pause 的 sandbox 在下次 acquire 时自动 resume（返回新句柄）。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    resumed_from: list[MockSandbox] = []

    async def resume_factory(old: MockSandbox) -> MockSandbox:
        resumed_from.append(old)
        new = MockSandbox(old.session_id)
        new.id = old.id + "-resumed"
        return new

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        sandbox_resume_factory=resume_factory,
        idle_pause_seconds=0.05,
        reaper_interval_seconds=0.02,
    )
    first = await pool.acquire("s1")
    # 等 reaper pause
    for _ in range(50):
        await asyncio.sleep(0.02)
        if pool.is_paused("s1"):
            break
    assert pool.is_paused("s1")

    # 下一次 acquire 应走 resume_factory 并返回新 sandbox
    second = await pool.acquire("s1")
    assert second is not first
    assert resumed_from == [first]
    assert not pool.is_paused("s1")
    await pool.close_all()


async def test_pool_active_sandbox_not_paused() -> None:
    """持续 acquire 更新 last_used_at，未达阈值前不应被 pause。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        idle_pause_seconds=0.5,  # 半秒阈值
        reaper_interval_seconds=0.02,
    )
    sb = await pool.acquire("s1")
    # 频繁 acquire 30 次 * 10ms = 300ms，全程 < 阈值
    for _ in range(30):
        await asyncio.sleep(0.01)
        await pool.acquire("s1")
    assert not sb.paused
    assert not pool.is_paused("s1")
    await pool.close_all()


async def test_pool_release_kills_paused_sandbox() -> None:
    """paused 状态下 release 仍应 kill 并清理 pause 标记。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        idle_pause_seconds=0.05,
        reaper_interval_seconds=0.02,
    )
    sb = await pool.acquire("s1")
    for _ in range(50):
        await asyncio.sleep(0.02)
        if pool.is_paused("s1"):
            break
    assert pool.is_paused("s1")
    await pool.release("s1")
    assert sb.killed
    assert not pool.has("s1")
    assert not pool.is_paused("s1")
    await pool.close_all()


async def test_pool_close_all_cancels_reaper() -> None:
    """close_all 取消 reaper task，不留悬挂 task。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        idle_pause_seconds=0.05,
        reaper_interval_seconds=0.02,
    )
    await pool.acquire("s1")
    assert pool._reaper_task is not None
    task = pool._reaper_task
    await pool.close_all()
    assert task.done()
    assert pool._reaper_task is None


async def test_pool_reaper_survives_pause_exception() -> None:
    """sandbox.pause 抛异常时 reaper 不应退出，后续 session 仍可被 pause。"""
    counter = {"n": 0}

    class FlakyPauseSandbox(MockSandbox):
        async def pause(self) -> None:
            counter["n"] += 1
            if counter["n"] == 1:
                raise RuntimeError("first pause fails")
            await super().pause()

    async def factory(session_id: str) -> FlakyPauseSandbox:
        return FlakyPauseSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        idle_pause_seconds=0.05,
        reaper_interval_seconds=0.02,
    )
    await pool.acquire("s1")
    await pool.acquire("s2")
    # 等多轮 reaper
    for _ in range(100):
        await asyncio.sleep(0.02)
        if pool.is_paused("s1") or pool.is_paused("s2"):
            # 至少一个成功 pause 了，并且 reaper 还活着
            break
    # 最终应至少有一个 session 被 pause 成功
    assert pool.is_paused("s1") or pool.is_paused("s2")
    assert pool._reaper_task is not None and not pool._reaper_task.done()
    await pool.close_all()


# ---------- tool source listing ----------

async def test_tool_source_lists_three_tools_with_default_prefix() -> None:
    src = OpenSandboxToolSource(make_pool(), fast_shell=False)
    tools = await src.list_tools()
    names = {t.name for t in tools}
    assert names == {"sandbox_shell", "sandbox_read_file", "sandbox_write_file"}


async def test_tool_source_custom_prefix() -> None:
    src = OpenSandboxToolSource(make_pool(), prefix="sb", fast_shell=False)
    tools = await src.list_tools()
    names = {t.name for t in tools}
    assert names == {"sb_shell", "sb_read_file", "sb_write_file"}


# ---------- shell handler ----------

async def test_shell_handler_success() -> None:
    src = OpenSandboxToolSource(make_pool(), fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    result = await tools["sandbox_shell"].handler(
        {"command": "echo hi"}, _ctx("s1")
    )
    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert "echo hi" in result["stdout"]


async def test_shell_handler_non_zero_exit() -> None:
    src = OpenSandboxToolSource(make_pool(), fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    result = await tools["sandbox_shell"].handler(
        {"command": "fail"}, _ctx("s1")
    )
    assert result["ok"] is False
    assert result["exit_code"] == 1
    assert "boom" in result["stderr"]


async def test_shell_handler_captures_exception() -> None:
    src = OpenSandboxToolSource(make_pool(), fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    result = await tools["sandbox_shell"].handler(
        {"command": "raise"}, _ctx("s1")
    )
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]


async def test_shell_handler_rejects_empty_command() -> None:
    src = OpenSandboxToolSource(make_pool(), fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    for bad in ("", "   ", None, 123):
        result = await tools["sandbox_shell"].handler(
            {"command": bad}, _ctx("s1")
        )
        assert result["ok"] is False


async def test_shell_handler_reuses_sandbox_within_session() -> None:
    pool = make_pool()
    src = OpenSandboxToolSource(pool, fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    await tools["sandbox_shell"].handler({"command": "echo 1"}, _ctx("s1"))
    await tools["sandbox_shell"].handler({"command": "echo 2"}, _ctx("s1"))
    sandbox = await pool.acquire("s1")
    assert sandbox.commands.calls == ["echo 1", "echo 2"]


# ---------- file handlers ----------

async def test_read_handler_rejects_relative_path() -> None:
    src = OpenSandboxToolSource(make_pool(), fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    result = await tools["sandbox_read_file"].handler(
        {"path": "relative.txt"}, _ctx("s1")
    )
    assert result["ok"] is False


async def test_read_handler_reports_missing_file() -> None:
    src = OpenSandboxToolSource(make_pool(), fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    result = await tools["sandbox_read_file"].handler(
        {"path": "/tmp/does-not-exist"}, _ctx("s1")
    )
    assert result["ok"] is False
    assert "FileNotFoundError" in result["error"]


async def test_write_handler_rejects_relative_path() -> None:
    src = OpenSandboxToolSource(make_pool(), write_entry_cls=MockWriteEntry, fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    result = await tools["sandbox_write_file"].handler(
        {"path": "relative.txt", "content": "x"}, _ctx("s1")
    )
    assert result["ok"] is False


async def test_write_handler_rejects_non_string_content() -> None:
    src = OpenSandboxToolSource(make_pool(), write_entry_cls=MockWriteEntry, fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    result = await tools["sandbox_write_file"].handler(
        {"path": "/tmp/x", "content": 123}, _ctx("s1")
    )
    assert result["ok"] is False


async def test_read_write_roundtrip() -> None:
    src = OpenSandboxToolSource(make_pool(), write_entry_cls=MockWriteEntry, fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    write_result = await tools["sandbox_write_file"].handler(
        {"path": "/tmp/x.txt", "content": "hello"}, _ctx("s1")
    )
    assert write_result["ok"] is True
    assert write_result["bytes_written"] == 5

    read_result = await tools["sandbox_read_file"].handler(
        {"path": "/tmp/x.txt"}, _ctx("s1")
    )
    assert read_result["ok"] is True
    assert read_result["content"] == "hello"


# ---------- fast_shell (httpx-direct SSE) ----------

async def test_fast_shell_happy_path() -> None:
    """正常路径：聚合 stdout，execution_complete 立即返回，exit_code=0."""
    def handler(_request):
        return httpx.Response(200, content=_sse_body(
            {"type": "init", "text": "exec-id"},
            {"type": "stdout", "text": "hello "},
            {"type": "stdout", "text": "world"},
            {"type": "execution_complete", "timestamp": 1},
        ))

    pool = make_fast_pool(handler)
    src = OpenSandboxToolSource(pool)  # 默认 fast_shell=True
    tools = {t.name: t for t in await src.list_tools()}
    try:
        result = await tools["sandbox_shell"].handler(
            {"command": "echo hello world"}, _ctx("s1")
        )
        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert result["stdout"] == "hello world"
        assert result["stderr"] == ""
        assert result["error"] is None
    finally:
        await pool.close_all()


async def test_fast_shell_early_break_after_complete() -> None:
    """execution_complete 之后即使还有 events，也不应计入结果（早退防 1s 阻塞）。"""
    def handler(_request):
        return httpx.Response(200, content=_sse_body(
            {"type": "stdout", "text": "seen"},
            {"type": "execution_complete", "timestamp": 1},
            {"type": "stdout", "text": "should-not-see"},
        ))

    pool = make_fast_pool(handler)
    src = OpenSandboxToolSource(pool)
    tools = {t.name: t for t in await src.list_tools()}
    try:
        result = await tools["sandbox_shell"].handler(
            {"command": "echo seen"}, _ctx("s1")
        )
        assert result["stdout"] == "seen"
        assert "should-not-see" not in result["stdout"]
    finally:
        await pool.close_all()


async def test_fast_shell_error_event() -> None:
    """error 事件的真实结构（execd v1.0.13）：error.evalue 是退出码。"""
    def handler(_request):
        return httpx.Response(200, content=_sse_body(
            {"type": "stderr", "text": "oops\n"},
            {"type": "error", "error": {
                "ename": "CommandExecError",
                "evalue": "2",
                "traceback": ["exit status 2"],
            }},
        ))

    pool = make_fast_pool(handler)
    src = OpenSandboxToolSource(pool)
    tools = {t.name: t for t in await src.list_tools()}
    try:
        result = await tools["sandbox_shell"].handler({"command": "false"}, _ctx("s1"))
        assert result["ok"] is False
        assert result["exit_code"] == 2
        assert result["stderr"] == "oops\n"
        assert result["error"] == "2"
    finally:
        await pool.close_all()


async def test_fast_shell_error_legacy_value_shape() -> None:
    """对 legacy `value` 字段保持兼容（防未来 execd 版本微调）。"""
    def handler(_request):
        return httpx.Response(200, content=_sse_body(
            {"type": "error", "value": "5"},
        ))

    pool = make_fast_pool(handler)
    src = OpenSandboxToolSource(pool)
    tools = {t.name: t for t in await src.list_tools()}
    try:
        result = await tools["sandbox_shell"].handler({"command": "x"}, _ctx("s1"))
        assert result["exit_code"] == 5
        assert result["error"] == "5"
    finally:
        await pool.close_all()


async def test_fast_shell_http_non_200() -> None:
    """execd 返回非 200 时：error 字段含状态码，不抛异常。"""
    def handler(_request):
        return httpx.Response(503, content=b"service unavailable")

    pool = make_fast_pool(handler)
    src = OpenSandboxToolSource(pool)
    tools = {t.name: t for t in await src.list_tools()}
    try:
        result = await tools["sandbox_shell"].handler({"command": "x"}, _ctx("s1"))
        assert result["ok"] is False
        assert result["error"] is not None
        assert "503" in result["error"]
    finally:
        await pool.close_all()


async def test_fast_shell_stderr_aggregation() -> None:
    """stderr 同 stdout 一样多片段拼接。"""
    def handler(_request):
        return httpx.Response(200, content=_sse_body(
            {"type": "stderr", "text": "err1 "},
            {"type": "stderr", "text": "err2"},
            {"type": "execution_complete", "timestamp": 1},
        ))

    pool = make_fast_pool(handler)
    src = OpenSandboxToolSource(pool)
    tools = {t.name: t for t in await src.list_tools()}
    try:
        result = await tools["sandbox_shell"].handler({"command": "x"}, _ctx("s1"))
        assert result["exit_code"] == 0
        assert result["stderr"] == "err1 err2"
    finally:
        await pool.close_all()


async def test_fast_shell_http_client_reused() -> None:
    """多次 tool 调用复用同一个 httpx.AsyncClient（连接池），pool 只创建一次。"""
    created = 0

    def handler(_request):
        return httpx.Response(200, content=_sse_body(
            {"type": "execution_complete", "timestamp": 1},
        ))

    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    def http_factory() -> httpx.AsyncClient:
        nonlocal created
        created += 1
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    pool = OpenSandboxPool(sandbox_factory=factory, http_client_factory=http_factory)
    src = OpenSandboxToolSource(pool)
    tools = {t.name: t for t in await src.list_tools()}
    try:
        for _ in range(5):
            await tools["sandbox_shell"].handler({"command": "x"}, _ctx("s1"))
        assert created == 1
    finally:
        await pool.close_all()


async def test_fast_shell_close_all_closes_http_client() -> None:
    """pool.close_all() 要 aclose 共享的 httpx.AsyncClient。"""
    closed = {"n": 0}

    class TrackedClient(httpx.AsyncClient):
        async def aclose(self, *args, **kwargs):
            closed["n"] += 1
            await super().aclose(*args, **kwargs)

    def handler(_request):
        return httpx.Response(200, content=_sse_body(
            {"type": "execution_complete", "timestamp": 1},
        ))

    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        http_client_factory=lambda: TrackedClient(transport=httpx.MockTransport(handler)),
    )
    src = OpenSandboxToolSource(pool)
    tools = {t.name: t for t in await src.list_tools()}
    await tools["sandbox_shell"].handler({"command": "x"}, _ctx("s1"))
    await pool.close_all()
    assert closed["n"] == 1


# ---------- file isolation (SDK path) ----------

async def test_files_are_isolated_across_sessions() -> None:
    """同路径在不同 session 下独立：s1 写入的文件 s2 读不到。"""
    src = OpenSandboxToolSource(make_pool(), write_entry_cls=MockWriteEntry, fast_shell=False)
    tools = {t.name: t for t in await src.list_tools()}
    await tools["sandbox_write_file"].handler(
        {"path": "/tmp/a.txt", "content": "from-s1"}, _ctx("s1")
    )
    result = await tools["sandbox_read_file"].handler(
        {"path": "/tmp/a.txt"}, _ctx("s2")
    )
    assert result["ok"] is False


# ---------- tenant binding + per-tenant quota ----------

async def test_bind_tenant_same_value_is_idempotent() -> None:
    pool = make_pool()
    pool.bind_tenant("s1", "tenant-a")
    pool.bind_tenant("s1", "tenant-a")  # no-op
    pool.bind_tenant("s2", None)
    pool.bind_tenant("s2", None)
    await pool.close_all()


async def test_bind_tenant_conflicting_rebind_raises() -> None:
    pool = make_pool()
    pool.bind_tenant("s1", "tenant-a")
    with pytest.raises(ValueError):
        pool.bind_tenant("s1", "tenant-b")
    await pool.close_all()


async def test_acquire_without_tenant_bypasses_quota() -> None:
    """未绑定 tenant 的 session 不走信号量，不受配额限制。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        per_tenant_max_sandboxes=1,
    )
    # 不传 tenant_id，连开 5 个 session 都不应被限制
    for i in range(5):
        await pool.acquire(f"s{i}")
    assert all(pool.has(f"s{i}") for i in range(5))
    await pool.close_all()


async def test_per_tenant_quota_blocks_until_timeout() -> None:
    """tenant 配额满且 timeout=0 时立即抛 TenantQuotaExceeded。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        per_tenant_max_sandboxes=2,
        per_tenant_acquire_timeout=0.05,
    )
    await pool.acquire("s1", tenant_id="t1")
    await pool.acquire("s2", tenant_id="t1")
    # 第 3 个超 t1 配额
    with pytest.raises(TenantQuotaExceeded) as exc_info:
        await pool.acquire("s3", tenant_id="t1")
    assert exc_info.value.tenant_id == "t1"
    assert exc_info.value.limit == 2
    await pool.close_all()


async def test_per_tenant_quota_released_on_release() -> None:
    """释放一个 session 腾出配额，下一个 acquire 成功。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        per_tenant_max_sandboxes=1,
        per_tenant_acquire_timeout=0.05,
    )
    await pool.acquire("s1", tenant_id="t1")
    with pytest.raises(TenantQuotaExceeded):
        await pool.acquire("s2", tenant_id="t1")
    await pool.release("s1")
    # 现在配额空了
    await pool.acquire("s2", tenant_id="t1")
    assert pool.has("s2")
    await pool.close_all()


async def test_per_tenant_quota_isolates_across_tenants() -> None:
    """租户 A 打满不影响租户 B。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        per_tenant_max_sandboxes=1,
        per_tenant_acquire_timeout=0.05,
    )
    await pool.acquire("a1", tenant_id="tenant-A")
    # tenant-A 已满，但 tenant-B 应该无感
    await pool.acquire("b1", tenant_id="tenant-B")
    assert pool.has("a1") and pool.has("b1")
    await pool.close_all()


async def test_factory_failure_does_not_consume_quota() -> None:
    """factory 抛异常时 tenant semaphore 必须 release，否则配额永久丢失。"""
    call_count = 0

    async def flaky_factory(session_id: str) -> MockSandbox:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=flaky_factory,
        per_tenant_max_sandboxes=1,
        per_tenant_acquire_timeout=0.05,
    )
    with pytest.raises(RuntimeError):
        await pool.acquire("s1", tenant_id="t1")
    # 配额没被消耗，立即可再 acquire
    await pool.acquire("s1", tenant_id="t1")
    await pool.close_all()


async def test_acquire_tenant_id_mismatch_raises() -> None:
    """session 已绑定 tenant-A，后续 acquire 传 tenant-B 应抛。"""
    pool = make_pool()
    await pool.acquire("s1", tenant_id="tenant-A")
    with pytest.raises(ValueError):
        await pool.acquire("s1", tenant_id="tenant-B")
    await pool.close_all()


# ---------- SessionSandboxBinding ----------

class _FakeEntry:
    """最小化 SessionEntry 替身：只暴露 session.tenant_id."""
    def __init__(self, tenant_id: str | None = None) -> None:
        self.session = type("S", (), {"tenant_id": tenant_id})()


async def test_binding_on_create_binds_tenant_from_entry() -> None:
    pool = make_pool()
    binding = SessionSandboxBinding(pool)
    entry = _FakeEntry(tenant_id="tenant-A")
    await binding.on_session_created("s1", entry)
    # 后续 acquire 不传 tenant_id 也应该命中已绑定的 tenant-A
    assert pool._tenant_of["s1"] == "tenant-A"
    await pool.close_all()


async def test_binding_on_create_handles_entry_without_tenant() -> None:
    """entry 没有 session 属性 / tenant_id 是 None 时 binding 不应抛。"""
    pool = make_pool()
    binding = SessionSandboxBinding(pool)
    await binding.on_session_created("s1", _FakeEntry(tenant_id=None))
    await binding.on_session_created("s2", object())  # 完全不符合 schema
    await pool.close_all()


async def test_binding_on_close_releases_sandbox() -> None:
    pool = make_pool()
    sb = await pool.acquire("s1")
    binding = SessionSandboxBinding(pool)
    await binding.on_session_closed("s1", _FakeEntry())
    assert sb.killed
    assert not pool.has("s1")
    await pool.close_all()


async def test_bind_tenant_prefix_applies_to_matching_sessions() -> None:
    """未显式 bind 的 session_id 以 prefix 开头时自动走 prefix 的 tenant。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=factory,
        per_tenant_max_sandboxes=1,
        per_tenant_acquire_timeout=0.05,
    )
    pool.bind_tenant_prefix("plan-abc:", "tenant-X")

    # 首次 acquire 以 plan-abc: 开头 → 自动绑 tenant-X + 消耗配额
    await pool.acquire("plan-abc:step-1:xxx")
    assert pool._tenant_of["plan-abc:step-1:xxx"] == "tenant-X"  # noqa: SLF001

    # tenant-X 配额已满，第二个 plan-abc 子 session 应被拒
    with pytest.raises(TenantQuotaExceeded):
        await pool.acquire("plan-abc:step-2:yyy")

    # 不以 plan-abc: 开头的 session 不受影响
    await pool.acquire("chat-sid")
    await pool.close_all()


async def test_bind_tenant_prefix_explicit_bind_wins() -> None:
    """同一 session_id 若已显式 bind，prefix 规则不覆盖。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(sandbox_factory=factory)
    pool.bind_tenant_prefix("p:", "tenant-prefix")
    pool.bind_tenant("p:explicit", "tenant-explicit")
    await pool.acquire("p:explicit")
    assert pool._tenant_of["p:explicit"] == "tenant-explicit"  # noqa: SLF001
    await pool.close_all()


async def test_bind_tenant_prefix_longest_match_wins() -> None:
    """多个 prefix 命中时取最长匹配。"""
    async def factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(sandbox_factory=factory)
    pool.bind_tenant_prefix("p:", "broad")
    pool.bind_tenant_prefix("p:team-a:", "specific")
    await pool.acquire("p:team-a:step-1")
    assert pool._tenant_of["p:team-a:step-1"] == "specific"  # noqa: SLF001
    await pool.acquire("p:team-b:step-1")
    assert pool._tenant_of["p:team-b:step-1"] == "broad"  # noqa: SLF001
    await pool.close_all()


async def test_bind_tenant_prefix_empty_raises() -> None:
    pool = make_pool()
    with pytest.raises(ValueError):
        pool.bind_tenant_prefix("", "t")
    await pool.close_all()


async def test_release_by_prefix_clears_prefix_binding() -> None:
    """release_by_prefix 应该同时从 _tenant_prefixes 里清理该 prefix。"""
    pool = make_pool()
    pool.bind_tenant_prefix("plan-x:", "tenant-A")
    assert "plan-x:" in pool._tenant_prefixes  # noqa: SLF001
    await pool.release_by_prefix("plan-x:")
    assert "plan-x:" not in pool._tenant_prefixes  # noqa: SLF001
    await pool.close_all()


async def test_binding_on_close_no_op_if_not_bound() -> None:
    """pool.release 幂等，对没绑过的 session 也不抛。"""
    pool = make_pool()
    binding = SessionSandboxBinding(pool)
    await binding.on_session_closed("never-existed", _FakeEntry())
    await pool.close_all()


# ---------- end-to-end: SessionStore + SessionSandboxBinding + quota ----------


async def test_session_store_wires_sandbox_lifecycle() -> None:
    """集成：SessionStore 创建 session → binding 通过 create hook 绑定 tenant →
    tool 调用走 pool.acquire 命中配额 → SessionStore.delete → pool.release。
    """
    from topsport_agent.server.sessions import SessionStore
    from topsport_agent.types.session import Session

    class FakeAgent:
        def __init__(self) -> None:
            self.closed = False

        def new_session(self, sid: str) -> Session:
            return Session(id=sid, system_prompt="")

        async def close(self) -> None:
            self.closed = True

    def agent_factory(_provider, _model):
        return FakeAgent()

    async def sandbox_factory(session_id: str) -> MockSandbox:
        return MockSandbox(session_id)

    pool = OpenSandboxPool(
        sandbox_factory=sandbox_factory,
        per_tenant_max_sandboxes=1,
        per_tenant_acquire_timeout=0.05,
    )
    binding = SessionSandboxBinding(pool)
    store = SessionStore(
        agent_factory=agent_factory,  # type: ignore[arg-type]
        provider=object(),  # type: ignore[arg-type]
        on_session_created=[binding.on_session_created],
        on_session_closed=[binding.on_session_closed],
    )

    # 创建租户 A 的两个 session（会被配额限制）
    sid_a1, entry_a1, is_new = await store.get_or_create(
        None, "mock/test", tenant_id="tenant-A", principal="alice"
    )
    assert is_new
    assert entry_a1.session.tenant_id == "tenant-A"
    assert entry_a1.session.principal == "alice"
    assert pool._tenant_of[sid_a1] == "tenant-A"  # create hook 已绑定

    # 首次 acquire 消耗 tenant-A 的 1 个配额
    await pool.acquire(sid_a1)

    # 第二个 session 属于同 tenant-A
    sid_a2, _, _ = await store.get_or_create(
        None, "mock/test", tenant_id="tenant-A", principal="alice"
    )
    with pytest.raises(TenantQuotaExceeded):
        await pool.acquire(sid_a2)

    # tenant-B 不受影响
    sid_b1, _, _ = await store.get_or_create(
        None, "mock/test", tenant_id="tenant-B", principal="bob"
    )
    await pool.acquire(sid_b1)
    assert pool.has(sid_b1)

    # delete sid_a1 触发 close hook → pool.release → tenant-A 配额腾出
    assert await store.delete(sid_a1)
    await pool.acquire(sid_a2)
    assert pool.has(sid_a2)

    await store.close_all()
    await pool.close_all()
    # close_all 后 binding 该被触发的都触发过
    assert not pool.has(sid_a2)
    assert not pool.has(sid_b1)
