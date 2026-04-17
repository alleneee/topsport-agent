from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .message import Message


class RunState(StrEnum):
    """状态机：IDLE -> RUNNING -> (WAITING_USER | WAITING_CONFIRM | DONE | ERROR)。

    WAITING_CONFIRM 用于需要人类审批的工具调用（如 shell 命令）。
    """
    IDLE = "idle"
    WAITING_USER = "waiting_user"
    WAITING_CONFIRM = "waiting_confirm"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass(slots=True)
class Session:
    """Session 是引擎的核心数据容器。

    messages 只存储实际对话历史，ContextProvider 的注入内容不写入 messages，避免每步重复膨胀。
    token_budget 设置后，每次 LLM_CALL_END 累加 usage 到 token_spent；超限时 Engine
    抛出 BudgetExceeded 并转为 RunState.ERROR —— 防止单个坏 prompt 烧掉自然语言赤字。
    """
    id: str
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    state: RunState = RunState.IDLE
    goal: str | None = None
    token_budget: int | None = None
    token_spent: int = 0
