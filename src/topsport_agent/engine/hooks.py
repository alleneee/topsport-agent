"""Engine 级与 Plan 级 hook 定义。

Engine Protocol 对应 ReAct 循环中不同阶段的扩展点。
Engine 通过 Protocol 与外部模块解耦：memory/skills/mcp/observability
都不直接被 engine 导入，而是在运行时注入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Union

from ..types.events import Event
from ..types.message import Message, ToolCall, ToolResult
from ..types.session import Session
from ..types.tool import ToolContext, ToolSpec

if TYPE_CHECKING:
    from ..types.plan import Plan, PlanStep, StepDecision
    from .orchestrator import SubAgentConfig


class ContextProvider(Protocol):
    """每步 LLM 调用前注入临时上下文，产出不落盘到 session.messages。"""

    name: str

    async def provide(self, session: Session) -> list[Message]: ...


class ToolSource(Protocol):
    """每步提供动态工具列表（如 MCP 桥接），按快照合并到工具池。"""

    name: str

    async def list_tools(self) -> list[ToolSpec]: ...


class PostStepHook(Protocol):
    """每步（含最后一步）结束后回调，用于 compaction / 状态持久化等。"""

    name: str

    async def after_step(self, session: Session, step: int) -> None: ...


class EventSubscriber(Protocol):
    """接收完整生命周期事件流，单个 subscriber 异常不影响引擎和其它 subscriber。

    critical 标记（可选，默认 False）—— True 表示该 subscriber 的失败是合规/审计
    层面的大事：引擎仍然不中断，但失败会被计入 Engine.subscriber_failures，
    便于外部健康检查读取、触发告警或拒绝继续接受新会话。
    """

    name: str

    async def on_event(self, event: Event) -> None: ...


# ---------------------------------------------------------------------------
# Pre/Post tool-use hooks (对标 Claude Code 的 PreToolUse / PostToolUse)
# 与现有 permission_checker / sanitizer 共存：permission 走专属决策路径
# （DENY/ALLOW/ASK 多阶段），sanitizer 走 PostToolUseHook 链的第一位。
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class HookAllow:
    """PreToolUseHook 放行。`updated_args` 非 None 时替换 call.arguments，
    多个 hook 链式时依次覆盖（最后一个胜出）。"""

    updated_args: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class HookDeny:
    """PreToolUseHook 拒绝。Engine 短路：写一个 is_error=True 的 ToolResult，
    跳过 handler 执行，仍走 audit + post hooks。"""

    reason: str


ToolDecision = Union[HookAllow, HookDeny]


class PreToolUseHook(Protocol):
    """Tool handler 调用前的扩展点。

    触发位置：permission_checker / permission_asker 已通过、handler.invoke 之前。
    多个 hook 按注册顺序执行；首个返回 HookDeny 立即短路，后续 hook 不再调用。
    HookAllow.updated_args 非 None 时改写后续 hook 与 handler 看到的 args
    （允许重写，例如路径净化）。

    异常处理：抛异常视为 hook 失败，Engine 写一条 warning，**不**短路链路
    （等同于 HookAllow 无 updated_args），让单个 hook 故障不阻塞工具执行。
    安全敏感的 hook 应自行包装为 HookDeny 而非抛异常。
    """

    name: str

    async def before_tool(
        self,
        call: ToolCall,
        tool: ToolSpec,
        ctx: ToolContext,
    ) -> ToolDecision: ...


class PostToolUseHook(Protocol):
    """Tool handler 返回之后、写入 session.messages 之前的扩展点。

    触发位置：handler.invoke 完成（含异常→is_error result）后立即触发；
    Engine 内置的 sanitizer 作为链首已运行（如配置）。`tool` 可能为 None
    （LLM 调用了未注册工具的 path），便于审计 hook 观测此类错误；安全/重写类
    hook 收到 None 时通常应原样返回。
    多个 hook 按注册顺序串成 result -> hook1 -> hook2 -> ... 链路；
    每个 hook 接收前一段输出，可改写或原样返回。

    异常处理：抛异常视为透传（log warning 后用前一段输出继续往后跑）。
    """

    name: str

    async def after_tool(
        self,
        call: ToolCall,
        tool: ToolSpec | None,
        result: ToolResult,
        ctx: ToolContext,
        *,
        trust_level: str,
    ) -> ToolResult: ...


# ---------------------------------------------------------------------------
# Plan 级 hook（Orchestrator 粒度）
# 补全 Engine hook 未覆盖的编排层：步骤配置定制和失败自动决策。
# 与 Engine hook 共同组成全链路干预体系：
#   StepConfigurator -> Engine hooks -> FailureHandler
# ---------------------------------------------------------------------------


class StepConfigurator(Protocol):
    """步骤执行前修改 sub-agent 配置：注入工具、切换模型、调整 system prompt。
    多个 configurator 按注册顺序链式执行，前一个的输出是后一个的输入。
    异常不中断流程——跳过该 configurator，沿用上一步的 config。
    """

    name: str

    async def configure_step(
        self, step: PlanStep, config: SubAgentConfig
    ) -> SubAgentConfig: ...


class FailureHandler(Protocol):
    """步骤失败时自动决策 retry/skip/abort，替代手动 provide_decision 等待。
    多个 handler 按注册顺序尝试，第一个成功返回的决策生效。
    全部失败或未注册时回退到 PLAN_WAITING 事件 + 外部手动决策。
    """

    name: str

    async def handle_failure(
        self, plan: Plan, failed_steps: list[PlanStep]
    ) -> StepDecision: ...
