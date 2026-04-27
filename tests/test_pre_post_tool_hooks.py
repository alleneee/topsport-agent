"""PreToolUseHook / PostToolUseHook 行为测试。

覆盖：
- pre 链顺序与 allow 透传
- pre HookDeny 短路 + 错误 ToolResult
- pre HookAllow.updated_args 改写后续 hook 与 handler 看到的 args
- pre hook 抛异常视为透传
- post 链按注册顺序串接 result 改写
- post hook 异常透传上一段 result
- post hook 仍触发 sanitizer 后（链首是 sanitizer 输出）
"""

from __future__ import annotations

import copy
from typing import Any

from topsport_agent.engine.hooks import HookAllow, HookDeny
from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.engine.sanitizer import DefaultSanitizer
from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.types.message import Role, ToolCall, ToolResult
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


class _Provider:
    name = "p"

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = list(turns)
        self._i = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        if self._i >= len(self._turns):
            return LLMResponse(text="end", finish_reason="stop")
        turn = self._turns[self._i]
        self._i += 1
        return turn


def _seen_args() -> dict[str, Any]:
    return {}


def _make_tool(seen: dict[str, Any], *, name: str = "echo", trust: str = "trusted") -> ToolSpec:
    async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        del ctx
        seen["args"] = dict(args)
        return {"echoed": args}

    return ToolSpec(
        name=name,
        description="",
        parameters={"type": "object"},
        handler=handler,
        trust_level=trust,
    )


def _session() -> Session:
    return Session(id="s", state=RunState.IDLE, system_prompt="")


def _two_turn(call_name: str = "echo", call_id: str = "c1") -> list[LLMResponse]:
    """First turn issues a tool call; second turn finishes with text."""
    return [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id=call_id, name=call_name, arguments={"x": 1})],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="done", finish_reason="stop"),
    ]


# ---------------------------------------------------------------------------
# PreToolUseHook
# ---------------------------------------------------------------------------


async def test_pre_hooks_run_in_registration_order_and_allow_passes_through() -> None:
    seen = _seen_args()
    order: list[str] = []

    class _PreA:
        name = "pre-a"

        async def before_tool(self, call, tool, ctx):
            order.append("a")
            return HookAllow()

    class _PreB:
        name = "pre-b"

        async def before_tool(self, call, tool, ctx):
            order.append("b")
            return HookAllow()

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[_make_tool(seen)],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_PreA(), _PreB()],
    )
    async for _ in eng.run(_session()):
        pass

    assert order == ["a", "b"]
    assert seen["args"] == {"x": 1}


async def test_pre_hook_deny_short_circuits_handler() -> None:
    seen = _seen_args()
    handler_invoked = {"flag": False}

    async def _h(args, ctx):
        del args, ctx
        handler_invoked["flag"] = True
        return {}

    tool = ToolSpec(
        name="echo", description="", parameters={"type": "object"}, handler=_h,
    )

    class _Deny:
        name = "deny"

        async def before_tool(self, call, tool, ctx):
            return HookDeny(reason="nope")

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[tool],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_Deny()],
    )
    sess = _session()
    async for _ in eng.run(sess):
        pass

    assert handler_invoked["flag"] is False
    # 工具结果（最后一条 TOOL 消息）应是 is_error 且 output=拒绝原因
    tool_msgs = [m for m in sess.messages if m.role == Role.TOOL]
    assert tool_msgs, "expected a tool result message"
    assert tool_msgs[-1].tool_results[0].is_error is True
    assert "nope" in str(tool_msgs[-1].tool_results[0].output)


async def test_pre_hook_allow_with_updated_args_rewrites_handler_input() -> None:
    seen = _seen_args()
    later_seen: dict[str, Any] = {}

    class _Rewrite:
        name = "rewrite"

        async def before_tool(self, call, tool, ctx):
            return HookAllow(updated_args={"x": 999, "added": True})

    class _Observe:
        name = "observe"

        async def before_tool(self, call, tool, ctx):
            # 链中第二个 hook 应看到 _Rewrite 改过的 args
            later_seen["seen"] = dict(call.arguments)
            return HookAllow()

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[_make_tool(seen)],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_Rewrite(), _Observe()],
    )
    async for _ in eng.run(_session()):
        pass

    assert seen["args"] == {"x": 999, "added": True}, "handler 应收到改写后的 args"
    assert later_seen["seen"] == {"x": 999, "added": True}, "下一段 hook 也应看到改写后的 args"


async def test_pre_hook_rewritten_args_get_revalidated() -> None:
    """HookAllow.updated_args 被改写后必须重跑 validate_input；改写出非法 args 应短路。"""
    handler_invoked = {"flag": False}

    async def _h(args, ctx):
        del args, ctx
        handler_invoked["flag"] = True
        return {}

    async def _validator(args: dict[str, Any]) -> str | None:
        if args.get("x") != 1:
            return f"x must be 1, got {args.get('x')!r}"
        return None

    tool = ToolSpec(
        name="echo",
        description="",
        parameters={"type": "object"},
        handler=_h,
        validate_input=_validator,
    )

    class _BadRewrite:
        name = "bad-rewrite"

        async def before_tool(self, call, tool, ctx):
            # 改写出违规的 args
            return HookAllow(updated_args={"x": 999})

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[tool],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_BadRewrite()],
    )
    sess = _session()
    async for _ in eng.run(sess):
        pass

    # validator 在改写后再跑应捕获非法输入并短路
    assert handler_invoked["flag"] is False, "改写出非法 args 时 handler 不应被调用"
    tool_msgs = [m for m in sess.messages if m.role == Role.TOOL]
    assert tool_msgs[-1].tool_results[0].is_error is True
    out = str(tool_msgs[-1].tool_results[0].output)
    assert "post-hook" in out and "x must be 1" in out


async def test_pre_hook_inplace_mutation_does_not_leak_upstream() -> None:
    """Hook 拿到的是 effective_args 的 deepcopy；嵌套对象的 in-place 突变也不
    影响 handler 看到的 args。守住"改写必须走 HookAllow(updated_args=...)"的契约。
    """
    seen = _seen_args()

    nested_seed = {"x": 1, "options": {"path": "/safe"}}

    async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        del ctx
        seen["args"] = copy.deepcopy(args)
        return {}

    tool = ToolSpec(
        name="echo",
        description="",
        parameters={"type": "object"},
        handler=handler,
    )

    # provider 推一次 tool_call，args 含嵌套 dict
    turns = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name="echo", arguments=nested_seed)],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="done", finish_reason="stop"),
    ]

    class _DeepMutate:
        name = "deep-mutate"

        async def before_tool(self, call, tool, ctx):
            # 嵌套层 in-place 突变
            call.arguments["options"]["path"] = "/etc/passwd"
            call.arguments["options"]["leaked"] = True
            call.arguments["x"] = 999
            return HookAllow()  # 没声明 updated_args

    eng = Engine(
        provider=_Provider(turns),
        tools=[tool],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_DeepMutate()],
    )
    async for _ in eng.run(_session()):
        pass

    assert seen["args"] == {"x": 1, "options": {"path": "/safe"}}, (
        "deepcopy 应阻止嵌套层 in-place mutate 泄漏到 handler"
    )


async def test_pre_hook_rewrite_makes_permission_check_see_final_args() -> None:
    """Permission 在 PreToolUseHook 之后跑，应对 hook 改写后的最终 args 把关。

    构造：原始 args="/safe"；hook 改写为"/etc/passwd"；checker 只在 path
    含 "/etc" 时返回 deny。期望：handler 不被调用。
    """

    async def handler(args, ctx):
        del args, ctx
        raise AssertionError("handler 不该被调用：permission 应基于 hook 改写后的 args 拒绝")

    tool = ToolSpec(
        name="read", description="",
        parameters={"type": "object"}, handler=handler,
    )

    class _Rewrite:
        name = "rewrite"

        async def before_tool(self, call, tool, ctx):
            return HookAllow(updated_args={"path": "/etc/passwd"})

    class _Checker:
        name = "etc-blocker"

        async def check(self, tool, call, ctx):
            del tool, ctx
            from topsport_agent.types.permission import deny, allow
            if "/etc" in str(call.arguments.get("path", "")):
                return deny(reason="etc denied")
            return allow()

    turns = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "/safe"})],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="done", finish_reason="stop"),
    ]
    eng = Engine(
        provider=_Provider(turns),
        tools=[tool],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_Rewrite()],
        permission_checker=_Checker(),
    )
    sess = _session()
    async for _ in eng.run(sess):
        pass

    tool_msgs = [m for m in sess.messages if m.role == Role.TOOL]
    out = str(tool_msgs[-1].tool_results[0].output)
    assert tool_msgs[-1].tool_results[0].is_error is True
    assert "etc denied" in out




async def test_pre_hook_exception_treated_as_passthrough() -> None:
    seen = _seen_args()

    class _Boom:
        name = "boom"

        async def before_tool(self, call, tool, ctx):
            raise RuntimeError("oops")

    class _Tail:
        name = "tail"

        async def before_tool(self, call, tool, ctx):
            return HookAllow()

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[_make_tool(seen)],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_Boom(), _Tail()],
    )
    async for _ in eng.run(_session()):
        pass

    # _Boom 抛异常但链路没断；handler 仍被调用
    assert seen["args"] == {"x": 1}


# ---------------------------------------------------------------------------
# PostToolUseHook
# ---------------------------------------------------------------------------


async def test_post_hooks_chain_in_order_and_can_rewrite_result() -> None:
    seen = _seen_args()

    class _Append:
        name = "append"

        async def after_tool(self, call, tool, result, ctx, *, trust_level):
            del call, tool, ctx, trust_level
            return ToolResult(
                call_id=result.call_id,
                output=str(result.output) + "::a",
                is_error=result.is_error,
            )

    class _Suffix:
        name = "suffix"

        async def after_tool(self, call, tool, result, ctx, *, trust_level):
            del call, tool, ctx, trust_level
            return ToolResult(
                call_id=result.call_id,
                output=str(result.output) + "::b",
                is_error=result.is_error,
            )

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[_make_tool(seen)],
        config=EngineConfig(model="m"),
        post_tool_hooks=[_Append(), _Suffix()],
    )
    sess = _session()
    async for _ in eng.run(sess):
        pass

    tool_msgs = [m for m in sess.messages if m.role == Role.TOOL]
    out = str(tool_msgs[-1].tool_results[0].output)
    # _Append 先跑，_Suffix 后跑：::a 在 ::b 之前
    assert out.endswith("::a::b"), f"expected chained output, got {out!r}"


async def test_post_hook_exception_passes_through_previous_result() -> None:
    seen = _seen_args()

    class _Boom:
        name = "boom"

        async def after_tool(self, call, tool, result, ctx, *, trust_level):
            del call, tool, ctx, trust_level, result
            raise RuntimeError("fail")

    class _Tail:
        name = "tail"

        async def after_tool(self, call, tool, result, ctx, *, trust_level):
            del call, tool, ctx, trust_level
            return ToolResult(
                call_id=result.call_id,
                output=str(result.output) + "::tail",
                is_error=result.is_error,
            )

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[_make_tool(seen)],
        config=EngineConfig(model="m"),
        post_tool_hooks=[_Boom(), _Tail()],
    )
    sess = _session()
    async for _ in eng.run(sess):
        pass

    tool_msgs = [m for m in sess.messages if m.role == Role.TOOL]
    out = str(tool_msgs[-1].tool_results[0].output)
    # _Boom 抛异常被吞，下一个 hook 收到 _Boom 之前的 result
    assert "::tail" in out


async def test_post_hook_runs_after_sanitizer() -> None:
    """sanitizer 在 post hook 链之前；hook 应观察到围栏后的 output。"""
    seen = _seen_args()
    captured: dict[str, str] = {}

    class _Inspect:
        name = "inspect"

        async def after_tool(self, call, tool, result, ctx, *, trust_level):
            del call, tool, ctx, trust_level
            captured["output"] = str(result.output)
            return result

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[_make_tool(seen, trust="untrusted")],
        config=EngineConfig(model="m"),
        sanitizer=DefaultSanitizer(),
        post_tool_hooks=[_Inspect()],
    )
    async for _ in eng.run(_session()):
        pass

    # untrusted 工具结果被 sanitizer 包了围栏，hook 看到的 output 应包含围栏 tag
    assert "<tool_output trust=\"untrusted\">" in captured["output"]


async def test_post_hook_sees_effective_args_after_pre_hook_rewrite() -> None:
    """post hook 的 call.arguments 必须是 handler 实际看到的最终 args，
    不是 LLM 提交的原 args（避免审计 hook 拿到陈旧输入）。"""
    seen: dict[str, Any] = {}

    async def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        del ctx
        return {}

    tool = ToolSpec(
        name="echo", description="",
        parameters={"type": "object"}, handler=handler,
    )

    class _Rewrite:
        name = "rewrite"

        async def before_tool(self, call, tool, ctx):
            return HookAllow(updated_args={"x": 999, "rewrote": True})

    class _Audit:
        name = "audit"

        async def after_tool(self, call, tool, result, ctx, *, trust_level):
            del tool, ctx, trust_level
            seen["call_args"] = dict(call.arguments)
            return result

    eng = Engine(
        provider=_Provider(_two_turn()),
        tools=[tool],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_Rewrite()],
        post_tool_hooks=[_Audit()],
    )
    async for _ in eng.run(_session()):
        pass

    assert seen["call_args"] == {"x": 999, "rewrote": True}, (
        "post hook 应观察到 PreToolUseHook 改写后的最终 args"
    )


async def test_extra_pre_post_hooks_propagate_through_agent_from_config() -> None:
    """通过 AgentConfig.extra_pre_tool_hooks / extra_post_tool_hooks 注册的 hook
    能在 from_config 构造的 Agent 上生效。"""
    from topsport_agent.agent import Agent, AgentConfig

    invoked = {"pre": False, "post": False}

    class _Pre:
        name = "pre"

        async def before_tool(self, call, tool, ctx):
            del call, tool, ctx
            invoked["pre"] = True
            return HookAllow()

    class _Post:
        name = "post"

        async def after_tool(self, call, tool, result, ctx, *, trust_level):
            del call, tool, ctx, trust_level
            invoked["post"] = True
            return result

    seen = _seen_args()
    cfg = AgentConfig(
        name="t", description="", system_prompt="", model="m",
        enable_skills=False, enable_memory=False, enable_plugins=False,
        enable_browser=False,
        extra_tools=[_make_tool(seen)],
        extra_pre_tool_hooks=[_Pre()],
        extra_post_tool_hooks=[_Post()],
    )
    agent = Agent.from_config(_Provider(_two_turn()), cfg)
    sess = agent.new_session()
    async for _ in agent.engine.run(sess):
        pass

    assert invoked["pre"] is True
    assert invoked["post"] is True


async def test_pre_hooks_serialize_across_concurrent_tool_calls() -> None:
    """LLM 同 turn 发多个 concurrency_safe 工具调用时，pre-hook 链跨调用必须
    串行执行（hook 没有并发安全契约）。Engine._pre_tool_hook_lock 守护此契约。
    """
    import asyncio

    seen = _seen_args()
    in_hook: dict[str, int] = {"count": 0, "max_concurrent": 0}

    async def _h(args, ctx):
        del args, ctx
        return {}

    tool = ToolSpec(
        name="echo",
        description="",
        parameters={"type": "object"},
        handler=_h,
        concurrency_safe=True,
    )

    class _Slow:
        name = "slow"

        async def before_tool(self, call, tool, ctx):
            del call, tool, ctx
            in_hook["count"] += 1
            in_hook["max_concurrent"] = max(in_hook["max_concurrent"], in_hook["count"])
            await asyncio.sleep(0.01)  # 给并发机会暴露
            in_hook["count"] -= 1
            return HookAllow()

    # provider 一次性发 3 个 tool_calls
    turns = [
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(id="c1", name="echo", arguments={"x": 1}),
                ToolCall(id="c2", name="echo", arguments={"x": 2}),
                ToolCall(id="c3", name="echo", arguments={"x": 3}),
            ],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="done", finish_reason="stop"),
    ]
    eng = Engine(
        provider=_Provider(turns),
        tools=[tool],
        config=EngineConfig(model="m"),
        pre_tool_hooks=[_Slow()],
    )
    del seen  # not used in this test
    async for _ in eng.run(_session()):
        pass

    # 锁保证跨 concurrent task 之间 pre-hook 链串行运行
    assert in_hook["max_concurrent"] == 1, (
        f"pre-hook 链应跨并发 tool calls 串行；观察到 max_concurrent="
        f"{in_hook['max_concurrent']}"
    )


async def test_post_hook_called_for_unregistered_tool_with_none_tool_arg() -> None:
    """LLM 调用未注册工具时 post hook 仍被触发，tool 参数为 None。"""
    seen_tool: dict[str, Any] = {}

    class _Observe:
        name = "observe"

        async def after_tool(self, call, tool, result, ctx, *, trust_level):
            del ctx, trust_level
            seen_tool["tool"] = tool
            seen_tool["call_name"] = call.name
            seen_tool["is_error"] = result.is_error
            return result

    # 提供 LLM 一个调用 unknown tool 的响应
    turns = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="c1", name="unknown_tool", arguments={})],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="done", finish_reason="stop"),
    ]

    eng = Engine(
        provider=_Provider(turns),
        tools=[],
        config=EngineConfig(model="m"),
        post_tool_hooks=[_Observe()],
    )
    async for _ in eng.run(_session()):
        pass

    assert seen_tool["tool"] is None
    assert seen_tool["call_name"] == "unknown_tool"
    assert seen_tool["is_error"] is True
