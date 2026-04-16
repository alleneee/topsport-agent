from __future__ import annotations

from typing import Protocol

from ..types.events import Event
from ..types.message import Message
from ..types.session import Session
from ..types.tool import ToolSpec


class ContextProvider(Protocol):
    name: str

    async def provide(self, session: Session) -> list[Message]: ...


class ToolSource(Protocol):
    name: str

    async def list_tools(self) -> list[ToolSpec]: ...


class PostStepHook(Protocol):
    name: str

    async def after_step(self, session: Session, step: int) -> None: ...


class EventSubscriber(Protocol):
    name: str

    async def on_event(self, event: Event) -> None: ...
