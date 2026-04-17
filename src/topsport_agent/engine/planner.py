from __future__ import annotations

import uuid
from typing import Any

from ..llm.provider import LLMProvider, StructuredOutputProvider
from ..llm.request import LLMRequest
from ..types.message import Message, Role
from ..types.plan import Plan, PlanStep
from ..types.tool import ToolSpec

# 借助 tool call 机制获取结构化输出：LLM 以工具参数形式返回计划，handler 永远不会被执行。
_PLAN_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique step identifier (kebab-case)",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short step title",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Detailed instructions for the sub-agent",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of steps that must complete first",
                    },
                },
                "required": ["id", "title", "instructions"],
            },
        },
    },
    "required": ["steps"],
}

PLAN_SYSTEM_PROMPT = (
    "You are a planning agent. Given a goal, create a plan with concrete steps.\n"
    "Each step will be executed by an independent sub-agent.\n"
    "Use depends_on to declare dependencies between steps.\n"
    "Steps without dependencies will run in parallel.\n"
    "Keep steps focused and independent. Each step should be self-contained."
)


class Planner:
    def __init__(self, provider: LLMProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    async def generate(
        self,
        goal: str,
        context: str = "",
        *,
        provider_options: dict[str, Any] | None = None,
    ) -> Plan:
        """H-A4: 优先走 StructuredOutputProvider（如果 provider 实现了），
        兜底才用 tool-call emulation。这让不支持 tool-use 的 provider（Gemini JSON
        mode、Bedrock、自建模型等）只要实现 complete_structured 就能跑计划。
        """
        # 系统提示 + 可选上下文 + 目标三段拼成完整输入。
        messages: list[Message] = []
        if context:
            messages.append(
                Message(role=Role.USER, content=f"Context:\n{context}")
            )
        messages.append(Message(role=Role.USER, content=f"Goal: {goal}"))

        request = LLMRequest(
            model=self._model,
            messages=[
                Message(role=Role.SYSTEM, content=PLAN_SYSTEM_PROMPT),
                *messages,
            ],
            tools=[],
            provider_options=dict(provider_options or {}),
        )

        if isinstance(self._provider, StructuredOutputProvider):
            arguments = await self._provider.complete_structured(
                request, _PLAN_TOOL_SCHEMA, tool_name="create_plan"
            )
            return _parse_plan(goal, arguments)

        return await self._generate_via_tool_call(goal, request)

    async def _generate_via_tool_call(self, goal: str, base: LLMRequest) -> Plan:
        """兜底路径：把 create_plan 塞进 tools 让 LLM 以 tool-call 形式返回 JSON。"""
        plan_tool = ToolSpec(
            name="create_plan",
            description="Create a structured execution plan with steps and dependencies",
            parameters=_PLAN_TOOL_SCHEMA,
            handler=_noop_handler,
        )
        request = LLMRequest(
            model=base.model,
            messages=list(base.messages),
            tools=[plan_tool],
            provider_options=dict(base.provider_options or {}),
        )
        response = await self._provider.complete(request)

        # 只取第一个 tool call；LLM 若返回多个则忽略后续的。
        if not response.tool_calls:
            raise ValueError("LLM did not return a plan tool call")

        call = response.tool_calls[0]
        if call.name != "create_plan":
            raise ValueError(f"Expected create_plan, got '{call.name}'")

        return _parse_plan(goal, call.arguments)


def _parse_plan(goal: str, arguments: dict[str, Any]) -> Plan:
    """Plan.__post_init__ 会触发 DAG 校验：自依赖、悬空依赖、环检测。"""
    raw_steps = arguments.get("steps", [])
    if not raw_steps:
        raise ValueError("Plan has no steps")
    steps = [
        PlanStep(
            id=s["id"],
            title=s["title"],
            instructions=s["instructions"],
            depends_on=s.get("depends_on", []),
        )
        for s in raw_steps
    ]
    return Plan(id=str(uuid.uuid4()), goal=goal, steps=steps)


async def _noop_handler(args: dict[str, Any], _ctx: Any) -> Any:
    """占位 handler，永远不会被调用——planner 直接截取 tool call 的参数作为结果。"""
    return args
