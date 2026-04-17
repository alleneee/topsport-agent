"""Engine 级与 Plan 级 hook 定义。

四个 Engine Protocol 分别对应 ReAct 循环中不同阶段的扩展点。
Engine 通过 Protocol 与外部模块解耦：memory/skills/mcp/observability
都不直接被 engine 导入，而是在运行时注入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..types.events import Event
from ..types.message import Message
from ..types.session import Session
from ..types.tool import ToolSpec

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
    """接收完整生命周期事件流，单个 subscriber 异常不影响引擎和其它 subscriber。"""

    name: str

    async def on_event(self, event: Event) -> None: ...


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
