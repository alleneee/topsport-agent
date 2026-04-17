from __future__ import annotations

from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .blob_store import BlobStore
from .output_cap import DEFAULT_CAP, enforce_cap


class ToolExecutor:
    """装饰器式执行器：在原始 handler 外层套上 output cap，引擎只需调用 wrap/wrap_all 即可。"""
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
        """返回新的 ToolSpec 而非修改原始对象，保持不可变语义。"""
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
