"""AgentIdentity / CapabilityToggles / CapabilityRegistry + AgentConfig
.identity / .toggles / .registry view round-trip 测试。

覆盖：
- 三个子 dataclass 独立构造（默认值合理 + 显式赋值）
- AgentConfig.from_parts 把三块拼回 flat 字段
- AgentConfig.identity / .toggles / .registry 视图返回的子 dataclass 值与
  flat 字段一致
- 修改视图 dataclass 不影响原 AgentConfig（每次返回新实例）
- 默认 AgentConfig 的视图值与子 dataclass 默认一致（语义对齐保护）
"""

from __future__ import annotations

import dataclasses

import pytest

from pathlib import Path

from topsport_agent.agent import (
    AgentConfig,
    AgentIdentity,
    CapabilityRegistry,
    CapabilityToggles,
)
from topsport_agent.agent.config_parts import (
    identity_field_names,
    registry_field_names,
    toggle_field_names,
)


def test_all_part_fields_default_values_match_agent_config_default() -> None:
    """循环检查所有 identity/toggles/registry 字段默认值与 AgentConfig
    flat 默认值一致。Loop-based 自动覆盖未来新增字段，避免 reviewer 指出的
    "只断言 8/18 字段，新增字段会静默漂移" 风险。"""
    cfg = AgentConfig()
    ident = AgentIdentity()
    togs = CapabilityToggles()
    reg = CapabilityRegistry()

    for n in identity_field_names():
        assert getattr(cfg, n) == getattr(ident, n), f"identity.{n} drift"
    for n in toggle_field_names():
        assert getattr(cfg, n) == getattr(togs, n), f"toggles.{n} drift"
    for n in registry_field_names():
        assert getattr(cfg, n) == getattr(reg, n), f"registry.{n} drift"


def test_part_field_names_partition_agent_config_fields_exactly() -> None:
    """三个 helper 返回的字段名并集应恰好等于 AgentConfig 的 flat 字段全集。
    避免新增 AgentConfig 字段时忘了归位到某个 part —— Pyright 看不出，
    必须靠运行时 schema guard 兜住。"""
    flat_fields = {f.name for f in dataclasses.fields(AgentConfig)}
    union = (
        set(identity_field_names())
        | set(toggle_field_names())
        | set(registry_field_names())
    )
    # AgentConfig 字段集应和三个 part 并集完全相等
    assert flat_fields == union, (
        f"part fields drift; AgentConfig has {flat_fields - union}, "
        f"part-only fields {union - flat_fields}"
    )
    # 三个 part 之间不应重叠
    assert set(identity_field_names()).isdisjoint(set(toggle_field_names()))
    assert set(identity_field_names()).isdisjoint(set(registry_field_names()))
    assert set(toggle_field_names()).isdisjoint(set(registry_field_names()))


def test_agent_config_identity_view_reflects_flat_fields() -> None:
    cfg = AgentConfig(
        name="alice", description="d", system_prompt="sp",
        model="claude-x", max_steps=42,
    )
    ident = cfg.identity
    assert ident.name == "alice"
    assert ident.description == "d"
    assert ident.system_prompt == "sp"
    assert ident.model == "claude-x"
    assert ident.max_steps == 42


def test_agent_config_toggles_view_reflects_flat_fields() -> None:
    p = Path("/tmp/m")
    cfg = AgentConfig(
        enable_skills=False, enable_memory=False, enable_browser=True,
        enable_file_ops=True, stream=True,
        memory_base_path=p, local_skill_dirs=[Path("/tmp/s")],
    )
    togs = cfg.toggles
    assert togs.enable_skills is False
    assert togs.enable_memory is False
    assert togs.enable_browser is True
    assert togs.enable_file_ops is True
    assert togs.stream is True
    assert togs.memory_base_path == p
    assert togs.local_skill_dirs == [Path("/tmp/s")]


def test_agent_config_registry_view_reflects_flat_fields() -> None:
    cfg = AgentConfig(
        tenant_id="t1", provider_options={"k": "v"},
    )
    reg = cfg.registry
    assert reg.tenant_id == "t1"
    assert reg.provider_options == {"k": "v"}


def test_view_returns_fresh_instances() -> None:
    """每次访问视图返回新实例（不缓存）。"""
    cfg = AgentConfig(enable_skills=True)
    togs1 = cfg.toggles
    togs2 = cfg.toggles
    assert togs1 is not togs2


def test_view_dataclasses_are_frozen_so_scalar_mutation_raises() -> None:
    """三个 view dataclass 都是 frozen=True：scalar 字段写入直接抛
    FrozenInstanceError，让"看起来能改其实没回写"的迷惑路径在运行时立刻暴露。"""
    togs = CapabilityToggles()
    with pytest.raises(dataclasses.FrozenInstanceError):
        togs.enable_skills = False  # type: ignore[misc]
    ident = AgentIdentity()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ident.name = "x"  # type: ignore[misc]
    reg = CapabilityRegistry()
    with pytest.raises(dataclasses.FrozenInstanceError):
        reg.tenant_id = "t"  # type: ignore[misc]


def test_view_list_is_isolated_from_agent_config_storage() -> None:
    """视图返回的 list/dict 是 shallow copy：mutating 视图的 list 不影响
    AgentConfig flat 字段。frozen 锁住 scalar 改写，_isolate 锁住 container 改写。"""
    cfg = AgentConfig(local_skill_dirs=[Path("/a")])
    togs = cfg.toggles
    togs.local_skill_dirs.append(Path("/b"))
    assert cfg.local_skill_dirs == [Path("/a")], "mutate view.list 不应回写 AgentConfig"

    cfg2 = AgentConfig(provider_options={"k": 1})
    reg2 = cfg2.registry
    reg2.provider_options["leaked"] = True  # type: ignore[index]
    assert cfg2.provider_options == {"k": 1}, "mutate view.dict 不应回写 AgentConfig"


def test_view_reflects_mutation_of_agent_config_flat_fields() -> None:
    """View 是按需 assemble 的快照：先取一次再改 flat 字段，下一次取视图反映改动。
    保证 view 不缓存"陈旧"快照。"""
    cfg = AgentConfig(enable_skills=True)
    snap_before = cfg.toggles
    cfg.enable_skills = False
    snap_after = cfg.toggles
    assert snap_before.enable_skills is True
    assert snap_after.enable_skills is False


def test_from_parts_isolates_input_lists_from_constructed_config() -> None:
    """from_parts(registry=CapabilityRegistry(extra_tools=[...])) 之后修改
    原 list，不应影响构造好的 AgentConfig（snapshot 语义对称：view 与构造
    路径都做 shallow copy）。"""
    seed_tools: list = []
    reg = CapabilityRegistry(extra_tools=seed_tools)
    cfg = AgentConfig.from_parts(registry=reg)
    seed_tools.append("contaminant")  # type: ignore[arg-type]
    assert cfg.extra_tools == [], "from_parts 应隔离输入 list"


def test_from_parts_round_trip() -> None:
    """from_parts(...) 拼出的 AgentConfig 字段值等于直接 flat 构造。"""
    cfg_a = AgentConfig.from_parts(
        identity=AgentIdentity(name="x", model="m1", max_steps=5),
        toggles=CapabilityToggles(enable_skills=False, enable_browser=True),
        registry=CapabilityRegistry(tenant_id="t", provider_options={"a": 1}),
    )
    cfg_b = AgentConfig(
        name="x", model="m1", max_steps=5,
        enable_skills=False, enable_browser=True,
        tenant_id="t", provider_options={"a": 1},
    )
    # 关键 flat 字段一致
    assert cfg_a.name == cfg_b.name
    assert cfg_a.model == cfg_b.model
    assert cfg_a.max_steps == cfg_b.max_steps
    assert cfg_a.enable_skills == cfg_b.enable_skills
    assert cfg_a.enable_browser == cfg_b.enable_browser
    assert cfg_a.tenant_id == cfg_b.tenant_id
    assert cfg_a.provider_options == cfg_b.provider_options


def test_from_parts_with_no_args_equals_default_config() -> None:
    """所有 part 都不传时，from_parts() 等价 AgentConfig() 默认。"""
    a = AgentConfig.from_parts()
    b = AgentConfig()
    assert a.name == b.name
    assert a.enable_skills == b.enable_skills
    assert a.tenant_id == b.tenant_id


def test_from_parts_partial_uses_defaults_for_missing_parts() -> None:
    """只传 identity 时，toggles 和 registry 走默认。"""
    cfg = AgentConfig.from_parts(identity=AgentIdentity(name="solo", model="m"))
    assert cfg.name == "solo"
    assert cfg.model == "m"
    # toggles 默认
    assert cfg.enable_skills is True
    assert cfg.enable_browser is False
    # registry 默认
    assert cfg.tenant_id is None
    assert cfg.extra_tools == []


def test_from_parts_round_trip_via_views() -> None:
    """from_parts(*c.views) 应产生与 c 等价的 config（伪幂等）。"""
    original = AgentConfig(
        name="n", model="m", max_steps=10,
        enable_skills=False, enable_memory=False, enable_browser=True,
        tenant_id="tx", provider_options={"a": 1},
    )
    rebuilt = AgentConfig.from_parts(
        identity=original.identity,
        toggles=original.toggles,
        registry=original.registry,
    )
    assert rebuilt.name == original.name
    assert rebuilt.model == original.model
    assert rebuilt.enable_skills == original.enable_skills
    assert rebuilt.enable_browser == original.enable_browser
    assert rebuilt.tenant_id == original.tenant_id
    assert rebuilt.provider_options == original.provider_options


def test_agent_from_parts_constructs_working_agent() -> None:
    """from_parts 的 AgentConfig 通过 Agent.from_config 能正常构造 Agent
    （没破任何下游 install pipeline）。"""
    from topsport_agent.agent import Agent
    from topsport_agent.llm.provider import LLMResponse
    from topsport_agent.llm.request import LLMRequest

    class _Provider:
        name = "p"

        async def complete(self, request: LLMRequest) -> LLMResponse:
            del request
            return LLMResponse(text="ok", finish_reason="stop")

    cfg = AgentConfig.from_parts(
        identity=AgentIdentity(name="t", model="m"),
        toggles=CapabilityToggles(
            enable_skills=False, enable_memory=False,
            enable_plugins=False, enable_browser=False,
        ),
    )
    agent = Agent.from_config(_Provider(), cfg)
    assert agent.config.name == "t"
    assert agent.config.model == "m"
