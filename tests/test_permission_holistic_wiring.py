"""Regression tests for the 4 holistic permission-wiring gaps codex flagged.

Each test corresponds to one specific gap in the capability-ACL integration:

1. `test_default_server_factory_honors_enable_gates`
   — server/app.py's default factory used to ignore enable_skills /
     enable_memory / enable_plugins; default_agent hardcoded them to True.
     Now each gate is honored end-to-end.

2. `test_session_store_runs_async_session_factory`
   — server/sessions.py used to call sync agent.new_session(), so persona
     resolution never happened in the server path. Now it always tries
     new_session_async first.

3. `test_spawn_child_inherits_permission_hooks_and_grants`
   — Agent.spawn_child used to drop permission_filter / audit_logger /
     permission_checker / permission_asker and the session's grants. Now
     all four hooks are forwarded and the sub_session inherits tenant /
     principal / granted_permissions / persona_id.

4. `test_namespaced_session_id_compatible_with_file_memory_store`
   — auth.namespace_session_id produces "principal::hint" but FileMemoryStore
     rejected ':' in path components — a default-chain landmine. Now ':' is
     whitelisted and the memory roundtrip works end-to-end.

5. `test_builtin_tools_declare_required_permissions`
   — file_ops / memory / agent_registry tools used to have empty
     required_permissions. Now they declare fs.read/fs.write/memory.write/
     agent.spawn so ToolVisibilityFilter can enforce the ACL.

6. `test_assignment_api_roundtrip_populates_session_grants`
   — Admin HTTP API previously had no /assignments endpoints, so the
     control plane could not bind personas to tenants. Now assignments
     drive the persona resolver hook in create_app.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from topsport_agent.agent.base import Agent, AgentConfig
from topsport_agent.engine.permission.assignment import (
    InMemoryAssignmentStore,
)
from topsport_agent.engine.permission.audit import (
    AuditLogger,
    InMemoryAuditStore,
)
from topsport_agent.engine.permission.filter import ToolVisibilityFilter
from topsport_agent.engine.permission.killswitch import KillSwitchGate
from topsport_agent.engine.permission.persona_registry import (
    InMemoryPersonaRegistry,
)
from topsport_agent.memory.file_store import FileMemoryStore
from topsport_agent.memory.tools import build_memory_tools
from topsport_agent.memory.types import MemoryEntry, MemoryType
from topsport_agent.plugins.agent_registry import (
    AgentRegistry,
    build_agent_tools,
)
from topsport_agent.server.app import _default_agent_factory, create_app
from topsport_agent.server.auth import namespace_session_id
from topsport_agent.server.config import ServerConfig
from topsport_agent.server.rbac import RBACPrincipal, _default_principal_resolver
from topsport_agent.tools.file_ops import file_tools
from topsport_agent.types.permission import (
    Permission,
    Persona,
    PersonaAssignment,
    Role,
)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeProvider:
    name = "fake"

    async def complete(self, request):
        from topsport_agent.llm.response import LLMResponse

        return LLMResponse(
            text="", tool_calls=[], finish_reason="end_turn",
            usage={}, response_metadata=None,
        )


# ---------------------------------------------------------------------------
# Gap 1: server default factory must honor every enable_* gate
# ---------------------------------------------------------------------------


def test_default_server_factory_honors_enable_gates():
    cfg = ServerConfig(
        auth_required=False,
        enable_file_tools=False,
        enable_skills=False,
        enable_memory=False,
        enable_plugins=False,
    )
    factory = _default_agent_factory(cfg)
    agent = factory(_FakeProvider(), "m")  # type: ignore[arg-type]
    report = agent.engine.capabilities_report()

    # None of file / skill / memory / plugin tools should leak through.
    names = set(report["tools"])
    assert "read_file" not in names
    assert "save_memory" not in names
    assert "recall_memory" not in names
    assert "load_skill" not in names
    assert "spawn_agent" not in names

    # And the agent-level capability flags are properly off.
    assert agent.config.enable_skills is False
    assert agent.config.enable_memory is False
    assert agent.config.enable_plugins is False


def test_default_server_factory_opens_only_flipped_gates():
    cfg = ServerConfig(
        auth_required=False,
        enable_file_tools=True,
        enable_skills=False,
        enable_memory=False,
        enable_plugins=False,
    )
    factory = _default_agent_factory(cfg)
    agent = factory(_FakeProvider(), "m")  # type: ignore[arg-type]
    names = set(agent.engine.capabilities_report()["tools"])

    assert "read_file" in names  # file tool gate is on
    assert "save_memory" not in names  # memory still off
    assert "spawn_agent" not in names  # plugins still off


# ---------------------------------------------------------------------------
# Gap 2: SessionStore must call agent.new_session_async (persona resolution)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_store_runs_async_session_factory():
    from topsport_agent.server.sessions import SessionStore

    dev = Persona(
        id="dev", display_name="Dev", description="",
        permissions=frozenset({Permission.FS_READ}),
    )
    registry = InMemoryPersonaRegistry()
    await registry.put(dev)

    def factory(provider, model):
        return Agent.from_config(
            _FakeProvider(),
            AgentConfig(
                model=model,
                persona="dev",
                persona_registry=registry,
                tenant_id="acme",
            ),
        )

    store = SessionStore(
        agent_factory=factory,  # type: ignore[arg-type]
        provider=_FakeProvider(),  # type: ignore[arg-type]
    )
    sid, entry, _ = await store.get_or_create(
        "sess-1", "m", tenant_id="acme", principal="alice",
    )
    # Persona grants must be populated via async path.
    assert Permission.FS_READ in entry.session.granted_permissions
    assert entry.session.persona_id == "dev"
    # Server-provided tenant/principal override agent config.
    assert entry.session.tenant_id == "acme"
    assert entry.session.principal == "alice"


# ---------------------------------------------------------------------------
# Gap 3: spawn_child must forward permission hooks + grants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_child_inherits_permission_hooks_and_grants():
    audit_store = InMemoryAuditStore()
    audit_logger = AuditLogger(store=audit_store)
    pfilter = ToolVisibilityFilter(audit_logger=audit_logger)

    dev = Persona(
        id="dev", display_name="Dev", description="",
        permissions=frozenset({Permission.FS_READ}),
    )

    agent = Agent.from_config(
        _FakeProvider(),
        AgentConfig(
            model="m",
            persona=dev,
            tenant_id="acme",
            permission_filter=pfilter,
            audit_logger=audit_logger,
        ),
    )

    # Seed the parent engine's running session so spawn_child can inherit
    # from it (simulates invocation from within a parent run).
    parent_session = await agent.new_session_async()
    parent_session.principal = "alice"
    agent._engine._current_session = parent_session  # type: ignore[attr-defined]

    sub_session, sub_engine = await agent.spawn_child(
        model="m", system_prompt="child", task="do something",
    )

    # Engine-level hooks are present.
    assert sub_engine._permission_filter is pfilter  # type: ignore[attr-defined]
    assert sub_engine._audit_logger is audit_logger  # type: ignore[attr-defined]

    # Session grants + identity propagated.
    assert Permission.FS_READ in sub_session.granted_permissions
    assert sub_session.tenant_id == "acme"
    assert sub_session.principal == "alice"
    assert sub_session.persona_id == "dev"


# ---------------------------------------------------------------------------
# Gap 4: session_id compatibility with FileMemoryStore
# ---------------------------------------------------------------------------


def test_namespaced_session_id_compatible_with_file_memory_store(tmp_path):
    # Reproduces codex's reported time-bomb scenario: server.auth generates
    # "principal::hint" session ids, and every memory inject step reads them.
    sid = namespace_session_id("anonymous", "demo")
    assert "::" in sid

    import asyncio

    store = FileMemoryStore(tmp_path)
    entry = MemoryEntry(
        key="k1", name="n", description="",
        type=MemoryType.NOTE, content="hello",
    )
    asyncio.run(store.write(sid, entry))
    roundtrip = asyncio.run(store.read(sid, "k1"))
    assert roundtrip is not None and roundtrip.content == "hello"


def test_file_memory_store_still_rejects_path_traversal(tmp_path):
    store = FileMemoryStore(tmp_path)
    import asyncio

    entry = MemoryEntry(
        key="k", name="n", description="",
        type=MemoryType.NOTE, content="x",
    )
    # '..' alone — always rejected.
    with pytest.raises(ValueError):
        asyncio.run(store.write("..", entry))
    # Traversal embedded in otherwise-valid charset — also rejected.
    with pytest.raises(ValueError):
        asyncio.run(store.write("a..b", entry))
    # Raw slash — rejected by regex.
    with pytest.raises(ValueError):
        asyncio.run(store.write("a/b", entry))


# ---------------------------------------------------------------------------
# Gap 5: builtin high-risk tools declare required_permissions
# ---------------------------------------------------------------------------


def test_builtin_file_tools_declare_required_permissions():
    tools = {t.name: t for t in file_tools()}
    assert tools["read_file"].required_permissions == frozenset({"fs.read"})
    assert tools["list_dir"].required_permissions == frozenset({"fs.read"})
    assert tools["glob_files"].required_permissions == frozenset({"fs.read"})
    assert tools["grep_files"].required_permissions == frozenset({"fs.read"})
    assert tools["write_file"].required_permissions == frozenset({"fs.write"})
    assert tools["edit_file"].required_permissions == frozenset({"fs.write"})


def test_builtin_memory_tools_declare_write_permission(tmp_path):
    store = FileMemoryStore(tmp_path)
    tools = {t.name: t for t in build_memory_tools(store)}
    assert tools["save_memory"].required_permissions == frozenset({"memory.write"})
    # recall_memory is read-only and intentionally has no required permission.
    assert tools["recall_memory"].required_permissions == frozenset()


def test_spawn_agent_tool_declares_agent_spawn_permission():
    registry = AgentRegistry()
    tools = {t.name: t for t in build_agent_tools(registry)}
    assert tools["spawn_agent"].required_permissions == frozenset({"agent.spawn"})
    # list_agents is a directory lookup — no permission needed.
    assert tools["list_agents"].required_permissions == frozenset()


# ---------------------------------------------------------------------------
# Gap 6: Assignment API bound to session-creation via resolver hook
# ---------------------------------------------------------------------------


def test_assignment_api_put_get_delete_roundtrip():
    from topsport_agent.server.permission_api import build_permission_router

    assignment_store = InMemoryAssignmentStore()
    app = FastAPI()
    router = build_permission_router(
        persona_registry=InMemoryPersonaRegistry(),
        audit_store=InMemoryAuditStore(),
        kill_switch=KillSwitchGate(),
        assignment_store=assignment_store,
    )
    app.include_router(router, prefix="/v1/admin")
    app.dependency_overrides[_default_principal_resolver] = (
        lambda: RBACPrincipal(user_id="admin", tenant_id="acme", role=Role.ADMIN)
    )
    client = TestClient(app)

    # PUT
    payload = {
        "tenant_id": "acme",
        "persona_ids": ["dev"],
        "default_persona_id": "dev",
        "user_id": "alice",
    }
    r = client.put("/v1/admin/assignments", json=payload)
    assert r.status_code == 200, r.text

    # GET
    r = client.get("/v1/admin/assignments", params={"tenant_id": "acme", "user_id": "alice"})
    assert r.status_code == 200
    body = r.json()
    assert body["default_persona_id"] == "dev"
    assert body["persona_ids"] == ["dev"]

    # DELETE
    r = client.delete(
        "/v1/admin/assignments",
        params={"tenant_id": "acme", "user_id": "alice"},
    )
    assert r.status_code == 200

    r = client.get("/v1/admin/assignments", params={"tenant_id": "acme", "user_id": "alice"})
    assert r.status_code == 404


def test_assignment_api_rejects_conflicting_user_and_group():
    from topsport_agent.server.permission_api import build_permission_router

    assignment_store = InMemoryAssignmentStore()
    app = FastAPI()
    router = build_permission_router(
        persona_registry=InMemoryPersonaRegistry(),
        audit_store=InMemoryAuditStore(),
        kill_switch=KillSwitchGate(),
        assignment_store=assignment_store,
    )
    app.include_router(router, prefix="/v1/admin")
    app.dependency_overrides[_default_principal_resolver] = (
        lambda: RBACPrincipal(user_id="admin", tenant_id="acme", role=Role.ADMIN)
    )
    client = TestClient(app)

    payload = {
        "tenant_id": "acme",
        "persona_ids": ["dev"],
        "default_persona_id": "dev",
        "user_id": "alice",
        "group_id": "team-a",
    }
    r = client.put("/v1/admin/assignments", json=payload)
    assert r.status_code == 400


def test_assignment_api_rejects_default_not_in_list():
    from topsport_agent.server.permission_api import build_permission_router

    assignment_store = InMemoryAssignmentStore()
    app = FastAPI()
    router = build_permission_router(
        persona_registry=InMemoryPersonaRegistry(),
        audit_store=InMemoryAuditStore(),
        kill_switch=KillSwitchGate(),
        assignment_store=assignment_store,
    )
    app.include_router(router, prefix="/v1/admin")
    app.dependency_overrides[_default_principal_resolver] = (
        lambda: RBACPrincipal(user_id="admin", tenant_id="acme", role=Role.ADMIN)
    )
    client = TestClient(app)

    r = client.put(
        "/v1/admin/assignments",
        json={
            "tenant_id": "acme",
            "persona_ids": ["dev"],
            "default_persona_id": "ops",
        },
    )
    assert r.status_code == 400


def test_assignment_api_returns_501_when_store_absent():
    from topsport_agent.server.permission_api import build_permission_router

    app = FastAPI()
    router = build_permission_router(
        persona_registry=InMemoryPersonaRegistry(),
        audit_store=InMemoryAuditStore(),
        kill_switch=KillSwitchGate(),
    )
    app.include_router(router, prefix="/v1/admin")
    app.dependency_overrides[_default_principal_resolver] = (
        lambda: RBACPrincipal(user_id="admin", tenant_id="acme", role=Role.ADMIN)
    )
    client = TestClient(app)

    r = client.put(
        "/v1/admin/assignments",
        json={"tenant_id": "acme", "persona_ids": []},
    )
    assert r.status_code == 501


@pytest.mark.asyncio
async def test_persona_resolver_hook_populates_grants_on_session_creation():
    # End-to-end control-plane ↔ execution-plane bridge:
    # persona + assignment pre-configured → create_app wires a create_hook →
    # SessionStore.get_or_create triggers it → session.granted_permissions set.
    registry = InMemoryPersonaRegistry()
    await registry.put(Persona(
        id="dev", display_name="Dev", description="",
        permissions=frozenset({Permission.FS_READ, Permission.MEMORY_WRITE}),
    ))
    assignment_store = InMemoryAssignmentStore()
    await assignment_store.put(PersonaAssignment(
        tenant_id="acme",
        persona_ids=frozenset({"dev"}),
        default_persona_id="dev",
        user_id="alice",
    ))

    cfg = ServerConfig(
        auth_required=False,
        enable_file_tools=True,  # enable some tools so we can verify filter
    )

    from topsport_agent.llm.provider import LLMProvider

    class _P:
        name = "fake"

        async def complete(self, request):
            from topsport_agent.llm.response import LLMResponse
            return LLMResponse(
                text="", tool_calls=[], finish_reason="end_turn",
                usage={}, response_metadata=None,
            )

    # Attach filter to the factory so we can assert filtering happens.
    pfilter = ToolVisibilityFilter()

    def agent_factory(provider: LLMProvider, model: str) -> Agent:
        return Agent.from_config(
            provider,
            AgentConfig(
                model=model,
                extra_tools=file_tools(),
                permission_filter=pfilter,
            ),
        )

    app = create_app(
        cfg,
        provider=_P(),  # type: ignore[arg-type]
        agent_factory=agent_factory,
        persona_registry=registry,
        audit_store=InMemoryAuditStore(),
        kill_switch=KillSwitchGate(),
        assignment_store=assignment_store,
    )

    with TestClient(app) as client:
        store = app.state.session_store
        sid, entry, _ = await store.get_or_create(
            "acme::alice", "m", tenant_id="acme", principal="alice",
        )
        # Hook populated grants from assignment → persona.
        assert entry.session.persona_id == "dev"
        assert Permission.FS_READ in entry.session.granted_permissions

        # Downstream filter now sees these grants and keeps fs.read tools.
        tools = await entry.agent.engine._snapshot_tools(entry.session)  # type: ignore[attr-defined]
        names = {t.name for t in tools}
        assert "read_file" in names
        # fs.write tool must be filtered out (dev persona lacks fs.write).
        assert "write_file" not in names


@pytest.mark.asyncio
async def test_persona_resolver_hook_failclosed_without_assignment():
    # No assignment matches → granted_permissions stays empty → any tagged
    # tool is invisible. Secure-by-default.
    registry = InMemoryPersonaRegistry()
    assignment_store = InMemoryAssignmentStore()

    cfg = ServerConfig(auth_required=False, enable_file_tools=True)

    class _P:
        name = "fake"

        async def complete(self, request):
            from topsport_agent.llm.response import LLMResponse
            return LLMResponse(
                text="", tool_calls=[], finish_reason="end_turn",
                usage={}, response_metadata=None,
            )

    pfilter = ToolVisibilityFilter()

    def agent_factory(provider, model):
        return Agent.from_config(
            provider,
            AgentConfig(
                model=model,
                extra_tools=file_tools(),
                permission_filter=pfilter,
            ),
        )

    app = create_app(
        cfg,
        provider=_P(),  # type: ignore[arg-type]
        agent_factory=agent_factory,
        persona_registry=registry,
        audit_store=InMemoryAuditStore(),
        kill_switch=KillSwitchGate(),
        assignment_store=assignment_store,
    )

    with TestClient(app) as client:
        store = app.state.session_store
        sid, entry, _ = await store.get_or_create(
            "nobody::x", "m", tenant_id="nobody", principal="x",
        )
        assert entry.session.granted_permissions == frozenset()
        tools = await entry.agent.engine._snapshot_tools(entry.session)  # type: ignore[attr-defined]
        names = {t.name for t in tools}
        # All tagged file tools must be invisible without grants.
        assert "read_file" not in names
        assert "write_file" not in names
