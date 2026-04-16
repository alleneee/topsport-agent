from __future__ import annotations

from typing import Protocol

from .request import LLMRequest
from .response import LLMResponse


class LLMProvider(Protocol):
    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
