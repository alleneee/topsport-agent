"""Permission 系统：checker + asker + Engine 集成。"""

from __future__ import annotations

import pytest

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.engine.permission import (
    AlwaysAskAsker,
    AlwaysDenyAsker,
    DefaultPermissionChecker,
)
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.message import Message, Role, ToolCall
from topsport_agent.types.permission import (
    PermissionBehavior,
    PermissionDecision,
    allow,
    ask,
    deny,
)
from topsport_agent.types.session import Session
from topsport_agent.types.tool import ToolContext, ToolSpec


class _Provider:
    name = "p"

    def __init__(self, rs: list[LLMResponse]) -> None:
        self._rs = list(rs)
        self._i = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        r = self._rs[self._i]
        self._i += 1
        return r


def _build_session() -> Session:
    s = Session(id="s", system_prompt="t")
    s.messages.append(Message(role=Role.USER, content="go"))
    return s


def _single_tool_call(name: str, args: dict | None = None) -> list[LLMResponse]:
    """剧本：LLM 调一次工具 → 调完 → 结束。"""
    return [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name=name, arguments=args or {})],
            finish_reason="tool_use", usage={}, response_metadata=None,
        ),
        LLMResponse(
            text="ok", tool_calls=[], finish_reason="end_turn",
            usage={}, response_metadata=None,
        ),
    ]


# ---------------------------------------------------------------------------
# Unit: PermissionDecision helpers
# ---------------------------------------------------------------------------


def test_allow_deny_ask_helpers():
    assert allow().behavior == PermissionBehavior.ALLOW
    assert deny("nope").behavior == PermissionBehavior.DENY
    assert deny("nope").reason == "nope"
    assert ask("verify").behavior == PermissionBehavior.ASK
    assert allow({"x": 1}).updated_input == {"x": 1}


def test_permission_decision_is_frozen():
    d = allow()
    with pytest.raises(Exception):
        d.behavior = PermissionBehavior.DENY  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Unit: DefaultPermissionChecker 策略
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_checker_destructive_requires_ask():
    async def h(a: dict, c: ToolContext) -> str:
        del a, c
        return "ok"

    destructive_tool = ToolSpec(name="rm", description="", parameters={}, handler=h,
                                destructive=True)
    checker = DefaultPermissionChecker()
    # ctx 用 None 占位——DefaultPermissionChecker 不访问
    decision = await checker.check(destructive_tool, ToolCall(id="c", name="rm", arguments={}), None)  # type: ignore[arg-type]
    assert decision.behavior == PermissionBehavior.ASK


@pytest.mark.asyncio
async def test_default_checker_read_only_allowed():
    async def h(a: dict, c: ToolContext) -> str:
        del a, c
        return "ok"

    read_tool = ToolSpec(name="read", description="", parameters={}, handler=h,
                         read_only=True)
    checker = DefaultPermissionChecker()
    decision = await checker.check(read_tool, ToolCall(id="c", name="read", arguments={}), None)  # type: ignore[arg-type]
    assert decision.behavior == PermissionBehavior.ALLOW


@pytest.mark.asyncio
async def test_default_checker_plain_tool_allowed():
    """既不 destructive 也不 read_only → ALLOW（向后兼容）。"""
    async def h(a: dict, c: ToolContext) -> str:
        del a, c
        return "ok"

    spec = ToolSpec(name="t", description="", parameters={}, handler=h)
    checker = DefaultPermissionChecker()
    decision = await checker.check(spec, ToolCall(id="c", name="t", arguments={}), None)  # type: ignore[arg-type]
    assert decision.behavior == PermissionBehavior.ALLOW


# ---------------------------------------------------------------------------
# Integration: Engine permission flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_checker_means_no_permission_checks():
    """没注入 checker → 跟改造前行为一致，destructive 工具也直接跑。"""
    ran = False

    async def h(a: dict, c: ToolContext) -> str:
        nonlocal ran
        ran = True
        del a, c
        return "done"

    spec = ToolSpec(name="rm", description="", parameters={}, handler=h, destructive=True)
    engine = Engine(_Provider(_single_tool_call("rm")), [spec], EngineConfig(model="m"))
    session = _build_session()
    async for _ in engine.run(session):
        pass
    assert ran is True


@pytest.mark.asyncio
async def test_destructive_denied_when_no_asker():
    """destructive + default checker + 无 asker → 保守默认 DENY。"""
    ran = False

    async def h(a: dict, c: ToolContext) -> str:
        nonlocal ran
        ran = True
        del a, c
        return "done"

    spec = ToolSpec(name="rm", description="", parameters={}, handler=h, destructive=True)
    engine = Engine(
        _Provider(_single_tool_call("rm")),
        [spec],
        EngineConfig(model="m"),
        permission_checker=DefaultPermissionChecker(),
        # permission_asker 故意不给
    )
    session = _build_session()
    async for _ in engine.run(session):
        pass

    assert ran is False, "handler must not run when permission denied"
    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    assert tool_msg.tool_results[0].is_error is True
    assert "destructive" in str(tool_msg.tool_results[0].output).lower()


@pytest.mark.asyncio
async def test_destructive_allowed_with_always_allow_asker():
    ran = False

    async def h(a: dict, c: ToolContext) -> str:
        nonlocal ran
        ran = True
        del a, c
        return "deleted"

    spec = ToolSpec(name="rm", description="", parameters={}, handler=h, destructive=True)
    engine = Engine(
        _Provider(_single_tool_call("rm")),
        [spec],
        EngineConfig(model="m"),
        permission_checker=DefaultPermissionChecker(),
        permission_asker=AlwaysAskAsker(),
    )
    session = _build_session()
    async for _ in engine.run(session):
        pass

    assert ran is True
    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    assert tool_msg.tool_results[0].output == "deleted"


@pytest.mark.asyncio
async def test_always_deny_asker_blocks_destructive():
    ran = False

    async def h(a: dict, c: ToolContext) -> str:
        nonlocal ran
        ran = True
        del a, c
        return "done"

    spec = ToolSpec(name="rm", description="", parameters={}, handler=h, destructive=True)
    engine = Engine(
        _Provider(_single_tool_call("rm")),
        [spec],
        EngineConfig(model="m"),
        permission_checker=DefaultPermissionChecker(),
        permission_asker=AlwaysDenyAsker(),
    )
    session = _build_session()
    async for _ in engine.run(session):
        pass
    assert ran is False


@pytest.mark.asyncio
async def test_updated_input_rewrites_args_before_handler():
    """checker 返回 ALLOW + updated_input → handler 收到改写后的参数。"""
    received: list[dict] = []

    async def h(a: dict, c: ToolContext) -> str:
        received.append(a)
        del c
        return "ok"

    spec = ToolSpec(name="t", description="", parameters={}, handler=h)

    class RewriteChecker:
        name = "rewrite"

        async def check(self, tool, call, context):
            del tool, context
            # 把 LLM 传的 path 扩展成绝对路径
            new_args = dict(call.arguments)
            new_args["path"] = "/abs/" + new_args.get("path", "")
            return allow(updated_input=new_args)

    engine = Engine(
        _Provider(_single_tool_call("t", {"path": "relative"})),
        [spec],
        EngineConfig(model="m"),
        permission_checker=RewriteChecker(),
    )
    session = _build_session()
    async for _ in engine.run(session):
        pass

    assert received == [{"path": "/abs/relative"}]


@pytest.mark.asyncio
async def test_custom_checker_deny_is_final():
    """不走 ASK 也能直接 DENY（比如 tenant 黑名单工具）。"""
    ran = False

    async def h(a: dict, c: ToolContext) -> str:
        nonlocal ran
        ran = True
        del a, c
        return "ok"

    spec = ToolSpec(name="forbidden", description="", parameters={}, handler=h)

    class BlacklistChecker:
        name = "blacklist"

        async def check(self, tool, call, context):
            del call, context
            if tool.name == "forbidden":
                return deny(f"tool {tool.name} is blacklisted for this tenant")
            return allow()

    engine = Engine(
        _Provider(_single_tool_call("forbidden")),
        [spec],
        EngineConfig(model="m"),
        permission_checker=BlacklistChecker(),
    )
    session = _build_session()
    async for _ in engine.run(session):
        pass

    assert ran is False
    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    assert "blacklisted" in str(tool_msg.tool_results[0].output)


@pytest.mark.asyncio
async def test_checker_exception_treated_as_deny():
    """checker 抛异常 → 按 DENY 处理（fail-closed 安全默认），不崩 engine。"""
    ran = False

    async def h(a: dict, c: ToolContext) -> str:
        nonlocal ran
        ran = True
        del a, c
        return "ok"

    spec = ToolSpec(name="t", description="", parameters={}, handler=h)

    class BrokenChecker:
        name = "broken"

        async def check(self, tool, call, context):
            del tool, call, context
            raise RuntimeError("checker crashed")

    engine = Engine(
        _Provider(_single_tool_call("t")),
        [spec],
        EngineConfig(model="m"),
        permission_checker=BrokenChecker(),
    )
    session = _build_session()
    async for _ in engine.run(session):
        pass

    assert ran is False
    tool_msg = next(m for m in session.messages if m.role == Role.TOOL)
    assert tool_msg.tool_results[0].is_error is True
    assert "checker error" in str(tool_msg.tool_results[0].output).lower()


@pytest.mark.asyncio
async def test_asker_can_override_checker_ask_to_allow_with_updated_input():
    """asker 也能通过 updated_input 改写参数——允许"sanitize after prompt"场景。"""
    received: list[dict] = []

    async def h(a: dict, c: ToolContext) -> str:
        received.append(a)
        del c
        return "ok"

    spec = ToolSpec(name="rm", description="", parameters={}, handler=h, destructive=True)

    class CleanupAsker:
        name = "cleanup"

        async def ask(self, tool, call, context, reason):
            del tool, call, context, reason
            return PermissionDecision(
                PermissionBehavior.ALLOW,
                updated_input={"path": "/safe/trash"},
            )

    engine = Engine(
        _Provider(_single_tool_call("rm", {"path": "/etc"})),
        [spec],
        EngineConfig(model="m"),
        permission_checker=DefaultPermissionChecker(),
        permission_asker=CleanupAsker(),
    )
    session = _build_session()
    async for _ in engine.run(session):
        pass

    assert received == [{"path": "/safe/trash"}]
