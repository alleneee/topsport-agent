from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ToolContext:
    """ToolContext 随每次工具调用创建，携带会话标识和取消信号。

    cancel_event 由 Engine.cancel() 触发，长时间运行的 handler 应周期性检查。
    workspace_root 为文件类工具提供沙箱边界：None 表示 CLI 信任模式不限制；
    设置后所有路径必须 resolve 落在该根目录内，防止符号链接逃逸。
    """
    session_id: str
    call_id: str
    cancel_event: asyncio.Event
    workspace_root: Path | None = None


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    """ToolSpec 是工具的完整描述：名称、JSON Schema 参数定义、异步 handler。

    引擎每步通过 _snapshot_tools 快照当前工具列表，不跨步缓存，保证动态工具源的实时性。
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
