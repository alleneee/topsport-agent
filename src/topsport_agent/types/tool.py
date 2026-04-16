from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ToolContext:
    session_id: str
    call_id: str
    cancel_event: asyncio.Event


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
