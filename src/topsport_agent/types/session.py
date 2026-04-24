from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from .message import Message

if TYPE_CHECKING:
    from ..workspace.manager import SessionWorkspace


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

    多租户字段（tenant_id、principal）可选；由 server 层在 SessionStore.get_or_create
    时注入，供工具层 / 观测层按租户维度做隔离、配额、审计。老路径（CLI、未传参）保持 None。
    """
    id: str
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    state: RunState = RunState.IDLE
    goal: str | None = None
    token_budget: int | None = None
    token_spent: int = 0
    tenant_id: str | None = None
    principal: str | None = None
    # Capability grants resolved from Persona at session creation.
    # Immutable for session lifetime; enforced by ToolVisibilityFilter.
    granted_permissions: frozenset[str] = field(default_factory=frozenset)
    # Persona id that populated granted_permissions (audit trail).
    persona_id: str | None = None
    # Per-session disk sandbox root (file_ops + potentially other disk-facing
    # tools). None = no sandbox → ToolContext.workspace_root stays None →
    # file_ops runs in CLI trust mode (host FS access). Production server
    # creates one per session via SessionStore lifecycle hook.
    workspace: "SessionWorkspace | None" = None
