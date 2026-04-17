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
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


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
