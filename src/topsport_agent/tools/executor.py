from __future__ import annotations

from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .blob_store import BlobStore
from .output_cap import DEFAULT_CAP, enforce_cap


class ToolExecutor:
    def __init__(
        self,
        *,
        caps: dict[str, int] | None = None,
        blob_store: BlobStore | None = None,
        default_cap: int = DEFAULT_CAP,
    ) -> None:
        self._caps = dict(caps or {})
        self._blob_store = blob_store
        self._default_cap = default_cap

    def wrap(self, spec: ToolSpec) -> ToolSpec:
        cap = self._caps.get(spec.name, self._default_cap)
        blob_store = self._blob_store
        original_handler = spec.handler

        async def capped_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            output = await original_handler(args, ctx)
            result = enforce_cap(output, cap, blob_store)
            return result.output

        return ToolSpec(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            handler=capped_handler,
        )

    def wrap_all(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        return [self.wrap(spec) for spec in specs]
