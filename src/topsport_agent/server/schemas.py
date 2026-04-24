"""OpenAI chat.completions 兼容的请求/响应模型 + Plan 执行模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI /v1/chat/completions 请求。

    - model: 必填 "provider/model-name" (anthropic/... 或 openai/...)
    - messages: 整段历史。服务端有状态时只取最后一条 user 消息作为新输入
    - stream: 是否走 SSE
    - user: OpenAI 标准字段，这里当作 session_id 使用（缺省随机生成）
    """

    model: str
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    # 请求级 system prompt 覆盖：非空则替换该 session 的 system_prompt
    # （持久到 session，和 OpenAI messages[0]=role:system 语义一致；两者同时给时
    # 该字段优先，便于多租户显式切换 persona）。
    system: str | None = None
    # Agent 执行模式。"react"（默认）= 单次 ReAct loop，消费 messages 里最后一条 user。
    # "plan" = DAG 分解执行，必须同时传 plan=<PlanSchema>，此时 messages 不强制。
    # 未来扩展（reflect / tree-of-thoughts 等）只在 Agent 内部加策略，不改 HTTP schema。
    mode: Literal["react", "plan"] = "react"
    plan: "PlanSchema | None" = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


class PlanStepSchema(BaseModel):
    id: str
    title: str
    instructions: str
    depends_on: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=1, ge=1, le=10)


class PlanSchema(BaseModel):
    id: str
    goal: str
    steps: list[PlanStepSchema]


class PlanExecuteRequest(BaseModel):
    plan: PlanSchema
    model: str
    max_steps: int = 10


class ErrorResponse(BaseModel):
    error: dict[str, Any]
