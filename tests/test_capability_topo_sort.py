"""CapabilityModule depends_on + topo-sort 行为测试。

覆盖：
- 无依赖时保持注册顺序（向后兼容老 list 顺序）
- 简单依赖 a -> b 让 a 先于 b 跑
- 多个 ready 同时就绪时按注册顺序 emit（稳定 tie-break）
- 钻石依赖 (a -> b/c -> d) 排序合法
- 未知依赖名 → CapabilityWiringError
- 循环依赖 → CapabilityWiringError
- 名字重复 → CapabilityWiringError
- 空输入 → 空输出
"""

from __future__ import annotations

import pytest

from topsport_agent.agent.capabilities import (
    CapabilityBundle,
    CapabilityModule,
    CapabilityWiringError,
    InstallContext,
    order_capability_modules,
)


class _StubModule:
    """最小 CapabilityModule 实现：runtime_checkable Protocol 通过即可。"""

    def __init__(self, name: str, depends_on: tuple[str, ...] = ()) -> None:
        self.name = name
        self.depends_on = depends_on

    def is_enabled(self, ctx: InstallContext) -> bool:
        del ctx
        return True

    def install(self, ctx: InstallContext) -> CapabilityBundle:
        del ctx
        return CapabilityBundle()


def _mod(name: str, depends_on: tuple[str, ...] = ()) -> CapabilityModule:
    return _StubModule(name, depends_on)


def _names(ordered: list[CapabilityModule]) -> list[str]:
    return [m.name for m in ordered]


def test_empty_input_returns_empty() -> None:
    assert order_capability_modules([]) == []


def test_no_dependencies_preserves_registration_order() -> None:
    ms = [_mod("a"), _mod("b"), _mod("c")]
    assert _names(order_capability_modules(ms)) == ["a", "b", "c"]


def test_simple_dependency_runs_dep_first() -> None:
    # b 依赖 a；即便 b 先注册也得 a 先跑
    ms = [_mod("b", ("a",)), _mod("a")]
    assert _names(order_capability_modules(ms)) == ["a", "b"]


def test_multiple_ready_modules_break_ties_on_registration_order() -> None:
    # a, b, c 均无依赖 → 同时 ready → 按注册顺序输出
    ms = [_mod("a"), _mod("b"), _mod("c")]
    assert _names(order_capability_modules(ms)) == ["a", "b", "c"]
    # 重排注册顺序，输出顺序应跟着重排
    ms2 = [_mod("c"), _mod("a"), _mod("b")]
    assert _names(order_capability_modules(ms2)) == ["c", "a", "b"]


def test_diamond_dependency_orders_correctly() -> None:
    # d 依赖 b 和 c；b 和 c 都依赖 a
    # 合法顺序：a, b, c, d 或 a, c, b, d
    ms = [
        _mod("d", ("b", "c")),
        _mod("c", ("a",)),
        _mod("b", ("a",)),
        _mod("a"),
    ]
    out = _names(order_capability_modules(ms))
    # a 必在 b 与 c 之前；d 必在 b 与 c 之后
    assert out.index("a") < out.index("b")
    assert out.index("a") < out.index("c")
    assert out.index("b") < out.index("d")
    assert out.index("c") < out.index("d")


def test_unknown_dependency_raises_wiring_error() -> None:
    ms = [_mod("a", ("missing",))]
    with pytest.raises(CapabilityWiringError) as excinfo:
        order_capability_modules(ms)
    assert "missing" in str(excinfo.value)
    assert "'a'" in str(excinfo.value)


def test_dependency_cycle_raises_wiring_error() -> None:
    # a -> b -> c -> a
    ms = [_mod("a", ("c",)), _mod("b", ("a",)), _mod("c", ("b",))]
    with pytest.raises(CapabilityWiringError) as excinfo:
        order_capability_modules(ms)
    assert "cycle" in str(excinfo.value).lower()


def test_duplicate_module_names_raises_wiring_error() -> None:
    ms = [_mod("a"), _mod("a")]
    with pytest.raises(CapabilityWiringError) as excinfo:
        order_capability_modules(ms)
    msg = str(excinfo.value)
    assert "duplicate" in msg.lower()
    assert "'a'" in msg
    # 新格式包含具体冲突位置，让 operator 知道哪里来的（典型场景：
    # extra_capability_modules 里的模块和某个默认模块同名）。
    assert "position 0" in msg
    assert "position 1" in msg


def test_skills_runs_after_plugins_via_default_modules() -> None:
    """SkillsModule 显式 depends_on=("plugins",) 在默认注册下应排在
    PluginsModule 之后。"""
    from topsport_agent.agent.capability_impls import default_capability_modules

    out = _names(order_capability_modules(default_capability_modules()))
    assert out.index("plugins") < out.index("skills")


def test_self_dependency_raises_with_module_name() -> None:
    """模块声明依赖自己时立即拒绝；错误信息必须包含肇事 module 名。
    防止退化到 graphlib.CycleError 路径产生 "unknown" 之类无信息错误。"""
    ms = [_mod("a", ("a",))]
    with pytest.raises(CapabilityWiringError) as excinfo:
        order_capability_modules(ms)
    msg = str(excinfo.value)
    assert "self-dependency" in msg
    assert "'a'" in msg


def test_cycle_error_message_contains_offending_node_names() -> None:
    """循环错误的提示必须能看到至少一个肇事节点名，operator 才能定位
    （之前 fallback 退化为 'unknown' 完全不可用）。"""
    ms = [_mod("a", ("c",)), _mod("b", ("a",)), _mod("c", ("b",))]
    with pytest.raises(CapabilityWiringError) as excinfo:
        order_capability_modules(ms)
    msg = str(excinfo.value)
    assert "cycle" in msg.lower()
    # 至少一个节点名出现在错误信息里
    assert any(name in msg for name in ("a", "b", "c"))


def test_duplicate_name_takes_precedence_over_missing_dependency() -> None:
    """两类错误同时出现时，duplicate name 先被检测（按 add 顺序扫一遍）。
    锁定语义：duplicate 是结构性 bug，比 missing dep 更优先。"""
    ms = [_mod("a", ("missing",)), _mod("a")]
    with pytest.raises(CapabilityWiringError) as excinfo:
        order_capability_modules(ms)
    assert "duplicate" in str(excinfo.value).lower()


def test_module_without_depends_on_attribute_is_treated_as_no_deps() -> None:
    """鸭子类型 module 不声明 depends_on 字段时按 () 处理，向后兼容。"""

    class _LegacyMod:
        # 不声明 depends_on
        name = "legacy"

        def is_enabled(self, ctx: InstallContext) -> bool:
            del ctx
            return True

        def install(self, ctx: InstallContext) -> CapabilityBundle:
            del ctx
            return CapabilityBundle()

    # _LegacyMod 缺 depends_on，仅靠 runtime fallback 正确处理；这里用 cast
    # 仅为安抚 Pyright，断言的是 runtime 行为。
    from typing import cast
    out = order_capability_modules([cast(CapabilityModule, _LegacyMod()), _mod("b")])
    assert _names(out) == ["legacy", "b"]


def test_extra_capability_modules_propagate_through_agent_from_config() -> None:
    """AgentConfig.extra_capability_modules 通过 Agent.from_config 进入 topo-sort
    并依据 depends_on 正确插队。"""
    from topsport_agent.agent import Agent, AgentConfig

    install_order: list[str] = []

    class _AfterMemory:
        name = "rag"
        depends_on: tuple[str, ...] = ("memory",)

        def is_enabled(self, ctx: InstallContext) -> bool:
            del ctx
            return True

        def install(self, ctx: InstallContext) -> CapabilityBundle:
            del ctx
            install_order.append("rag")
            return CapabilityBundle()

    class _Provider:
        name = "p"

        async def complete(self, request):  # type: ignore[no-untyped-def]
            del request
            from topsport_agent.llm.provider import LLMResponse
            return LLMResponse(text="ok", finish_reason="stop")

    cfg = AgentConfig(
        name="t", description="", system_prompt="", model="m",
        enable_skills=False, enable_plugins=False, enable_browser=False,
        enable_memory=True,
        extra_capability_modules=[_AfterMemory()],
    )
    Agent.from_config(_Provider(), cfg)
    # rag 真的被 install 了（在 memory 之后跑成功，没被 unknown-dep 拒掉）
    assert install_order == ["rag"]
