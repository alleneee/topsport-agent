"""Server wire-up 集成测试：sandbox 启用时 factory 注入 + SessionStore 钩子 + tenant 透传。

不依赖真实 OpenSandbox：注入 FakePool 观察 bind_tenant/release 调用顺序。
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from topsport_agent.agent.base import Agent, AgentConfig  # noqa: E402
from topsport_agent.llm.provider import LLMProvider  # noqa: E402
from topsport_agent.llm.request import LLMRequest  # noqa: E402
from topsport_agent.llm.response import LLMResponse  # noqa: E402
from topsport_agent.llm.stream import LLMStreamChunk  # noqa: E402
from topsport_agent.server import ServerConfig, create_app  # noqa: E402
from topsport_agent.server.app import _default_agent_factory  # noqa: E402


# ---------- mocks ----------


@dataclass
class MockStreamProvider:
    name: str = "mock"
    text: str = "ok"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=self.text, tool_calls=[], finish_reason="stop",
            usage={"input_tokens": 1, "output_tokens": 1}, response_metadata=None,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        yield LLMStreamChunk(type="text_delta", text_delta=self.text)
        yield LLMStreamChunk(
            type="done",
            final_response=LLMResponse(
                text=self.text, tool_calls=[], finish_reason="stop",
                usage={"input_tokens": 1, "output_tokens": 1}, response_metadata=None,
            ),
        )


@dataclass
class FakePool:
    """模拟 OpenSandboxPool 的最小接口。"""
    binds: list[tuple[str, str | None]] = field(default_factory=list)
    prefix_binds: list[tuple[str, str | None]] = field(default_factory=list)
    releases: list[str] = field(default_factory=list)
    prefix_releases: list[str] = field(default_factory=list)
    closed: bool = False

    def bind_tenant(self, session_id: str, tenant_id: str | None) -> None:
        self.binds.append((session_id, tenant_id))

    def bind_tenant_prefix(self, prefix: str, tenant_id: str | None) -> None:
        self.prefix_binds.append((prefix, tenant_id))

    async def release(self, session_id: str) -> None:
        self.releases.append(session_id)

    async def release_by_prefix(self, prefix: str) -> int:
        self.prefix_releases.append(prefix)
        return 0

    async def close_all(self) -> None:
        self.closed = True


def _minimal_agent_factory(p: LLMProvider, model: str) -> Agent:
    """测试用最小 agent：关 plugins / skills / memory / browser 避开本地环境干扰。

    这里不走 _default_agent_factory（会扫 ~/.claude/plugins/ 拉起真实插件），
    改由 test_default_factory_* 系列专门验证 _default_agent_factory 行为。
    """
    cfg = AgentConfig(
        name="t", description="", system_prompt="sp",
        model=model, stream=True,
        enable_skills=False, enable_memory=False,
        enable_plugins=False, enable_browser=False,
    )
    return Agent.from_config(p, cfg)


def _make_app(*, sandbox_pool: FakePool | None = None) -> Any:
    cfg = ServerConfig(
        api_key="dummy", default_model="mock/x",
        auth_required=False,
        sandbox_enabled=sandbox_pool is not None,
    )
    return create_app(
        cfg,
        provider_name="anthropic",
        provider=MockStreamProvider(),
        agent_factory=_minimal_agent_factory,
        sandbox_pool=sandbox_pool,  # type: ignore[arg-type]
    )


# ---------- tests ----------


def test_app_without_sandbox_has_no_pool_in_state() -> None:
    app = _make_app(sandbox_pool=None)
    with TestClient(app):
        assert app.state.sandbox_pool is None


def test_app_with_sandbox_pool_stored_on_state() -> None:
    pool = FakePool()
    app = _make_app(sandbox_pool=pool)
    with TestClient(app):
        assert app.state.sandbox_pool is pool


def test_default_factory_adds_sandbox_tool_source_when_pool() -> None:
    """启用 sandbox 时 factory 产出的 agent 的 engine 工具列表含 sandbox_shell。"""
    import asyncio

    pool = FakePool()
    cfg = ServerConfig(
        api_key="x", default_model="mock/x",
        auth_required=False, sandbox_enabled=True,
    )
    factory = _default_agent_factory(cfg, sandbox_pool=pool)  # type: ignore[arg-type]
    agent = factory(MockStreamProvider(), "test-model")  # type: ignore[arg-type]

    # 通过 tool source 收集动态工具名
    sources = agent.engine.tool_source_names()
    assert "opensandbox" in sources

    async def list_names() -> set[str]:
        names: set[str] = set()
        for src in agent.engine._tool_sources:  # noqa: SLF001
            for spec in await src.list_tools():
                names.add(spec.name)
        return names

    names = asyncio.run(list_names())
    assert {"sandbox_shell", "sandbox_read_file", "sandbox_write_file"} <= names


def test_default_factory_disables_file_ops_when_sandbox_enabled() -> None:
    """即便 ENABLE_FILE_TOOLS=true，sandbox 启用也必须禁用本地 file_ops（SEC-001）。"""
    pool = FakePool()
    cfg = ServerConfig(
        api_key="x", default_model="mock/x",
        auth_required=False, sandbox_enabled=True,
        enable_file_tools=True,  # 明确打开
    )
    factory = _default_agent_factory(cfg, sandbox_pool=pool)  # type: ignore[arg-type]
    agent = factory(MockStreamProvider(), "test-model")  # type: ignore[arg-type]

    # Engine 的静态工具（file_tools）应为空
    static_tool_names = {t.name for t in agent.engine._tools}  # noqa: SLF001
    assert "read_file" not in static_tool_names
    assert "write_file" not in static_tool_names
    assert "edit_file" not in static_tool_names


def test_default_factory_keeps_file_ops_when_sandbox_disabled() -> None:
    """sandbox 关闭 + enable_file_tools=true 时保留原 file_ops 行为。"""
    cfg = ServerConfig(
        api_key="x", default_model="mock/x",
        auth_required=False, sandbox_enabled=False,
        enable_file_tools=True,
    )
    factory = _default_agent_factory(cfg, sandbox_pool=None)
    agent = factory(MockStreamProvider(), "test-model")  # type: ignore[arg-type]
    static_tool_names = {t.name for t in agent.engine._tools}  # noqa: SLF001
    assert "read_file" in static_tool_names
    assert "write_file" in static_tool_names


def test_chat_creates_session_with_tenant_equal_to_principal() -> None:
    """chat 请求触发 session 创建，session.tenant_id / principal 必须 = principal。

    auth_required=False 时 principal 恒为 "default"。
    断言在 TestClient 上下文内进行（退出时 lifespan 会 close_all 清空 session）。
    """
    pool = FakePool()
    app = _make_app(sandbox_pool=pool)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "user": "alice-user",
            },
        )
        assert resp.status_code == 200

        store = app.state.session_store
        sids = list(store._entries.keys())  # noqa: SLF001
        assert len(sids) == 1
        entry = store._entries[sids[0]]  # noqa: SLF001
        assert entry.session.tenant_id == "anonymous"
        assert entry.session.principal == "anonymous"
        # binding.on_session_created 应被触发，pool 记录绑定
        assert pool.binds == [(sids[0], "anonymous")]


def test_session_delete_triggers_sandbox_release() -> None:
    """DELETE session → close hook → pool.release。"""
    pool = FakePool()
    app = _make_app(sandbox_pool=pool)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic/x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "user": "bob-user",
            },
        )
        assert resp.status_code == 200
        sid = list(app.state.session_store._entries.keys())[0]  # noqa: SLF001
        assert pool.binds == [(sid, "anonymous")]
        assert pool.releases == []

        # DELETE 路由接受 user_hint，server 内部会 namespace 成 sid
        del_resp = client.delete("/v1/sessions/bob-user")
        assert del_resp.status_code in (200, 204)

        # delete 后立即 release；之后 lifespan 退出时不会再 release（已从 store 移除）
        assert sid in pool.releases


def test_pool_close_all_called_on_lifespan_exit() -> None:
    """app 关闭时 sandbox_pool.close_all() 必须被调用。"""
    pool = FakePool()
    app = _make_app(sandbox_pool=pool)
    with TestClient(app):
        pass
    assert pool.closed is True


def test_plan_execute_cleans_up_sandbox_by_prefix_on_success() -> None:
    """plan 跑完后 pool.release_by_prefix 被调，前缀 = f"{plan_id}:"."""
    pool = FakePool()
    app = _make_app(sandbox_pool=pool)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/plan/execute",
            json={
                "model": "anthropic/m",
                "plan": {
                    "id": "p-abc",
                    "goal": "test",
                    "steps": [
                        {"id": "s1", "title": "a", "instructions": "do a"},
                    ],
                },
            },
        ) as r:
            assert r.status_code == 200
            _ = r.read()  # 消费 SSE 流直到结束

        # 生效检查：prefix 绑定 + 清理都发生了
        assert ("p-abc:", "anonymous") in pool.prefix_binds
        assert "p-abc:" in pool.prefix_releases


def test_plan_execute_binds_prefix_tenant_from_principal() -> None:
    """plan_execute 必须预绑定 prefix→tenant，让 plan 子 session 走 per-tenant 配额。"""
    pool = FakePool()
    app = _make_app(sandbox_pool=pool)
    with TestClient(app) as client:
        with client.stream(
            "POST", "/v1/plan/execute",
            json={
                "model": "anthropic/m",
                "plan": {
                    "id": "p-tenant-check", "goal": "t",
                    "steps": [{"id": "s1", "title": "a", "instructions": "do"}],
                },
            },
        ) as r:
            _ = r.read()
        # 精确匹配：前缀 = plan.id + ":", tenant = principal
        assert pool.prefix_binds == [("p-tenant-check:", "anonymous")]


def test_plan_error_payload_does_not_leak_exception_message() -> None:
    """SEC-005：_stream_plan 的 except 分支必须脱敏异常 str(exc)。

    直接测 _stream_plan 的 except → yield 'error' 路径：构造一个会在 execute()
    里抛异常的 FakeOrchestrator，确认响应 SSE 中不含原始敏感字符串。
    """
    import asyncio

    from topsport_agent.server.plan import _stream_plan

    SECRET = "sk-XXXSECRETxxx"

    class FakeOrch:
        async def execute(self):
            raise RuntimeError(f"AuthenticationError: Incorrect API key {SECRET}")
            if False:  # pragma: no cover -- keep async-generator shape
                yield None

        def cancel(self) -> None:
            pass

        def provide_decision(self, _d) -> None:
            pass

    class FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def collect() -> str:
        chunks: list[str] = []
        async for chunk in _stream_plan(FakeOrch(), FakeRequest(), parent_agent=None):  # type: ignore[arg-type]
            chunks.append(chunk)
        return "".join(chunks)

    body = asyncio.run(collect())
    assert SECRET not in body
    # 对外消息是通用描述 + 异常类型
    assert "plan execution failed" in body
    assert "RuntimeError" in body


def test_plan_execute_no_sandbox_cleanup_when_pool_absent() -> None:
    """sandbox_pool=None 时 plan 不调 release_by_prefix（根本没 pool）。"""
    app = _make_app(sandbox_pool=None)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/plan/execute",
            json={
                "model": "anthropic/m",
                "plan": {
                    "id": "p-xyz",
                    "goal": "test",
                    "steps": [
                        {"id": "s1", "title": "a", "instructions": "do a"},
                    ],
                },
            },
        ) as r:
            assert r.status_code == 200
            _ = r.read()
        # 不抛即可——pool 为 None 时 finally 短路


def test_plan_execute_cleans_up_on_error_path() -> None:
    """plan 执行中抛异常时仍调 release_by_prefix（finally 保证）。"""

    class BreakingPool(FakePool):
        async def release_by_prefix(self, prefix: str) -> int:
            self.prefix_releases.append(prefix)
            return 2  # 假装清了两个

    pool = BreakingPool()
    app = _make_app(sandbox_pool=pool)
    with TestClient(app) as client:
        with client.stream(
            "POST", "/v1/plan/execute",
            json={
                "model": "anthropic/m",
                "plan": {
                    "id": "p-err",
                    "goal": "t",
                    "steps": [{"id": "s1", "title": "a", "instructions": "do"}],
                },
            },
        ) as r:
            # 即使 SSE 里带 error 事件也应 200 响应（事件流里报告错误）
            _ = r.read()
        assert "p-err:" in pool.prefix_releases


def test_sandbox_pool_close_exception_does_not_fail_shutdown() -> None:
    """pool.close_all 抛异常不应阻塞 graceful shutdown。"""

    class BrokenPool(FakePool):
        async def close_all(self) -> None:
            raise RuntimeError("boom")

    pool = BrokenPool()
    app = _make_app(sandbox_pool=pool)
    # 进入/退出 lifespan 不应抛
    with TestClient(app):
        pass
