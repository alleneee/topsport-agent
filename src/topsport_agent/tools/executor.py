from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .blob_store import BlobStore
from .output_cap import DEFAULT_CAP, enforce_cap


class ToolExecutor:
    """装饰器式执行器：在原始 handler 外层套上 output cap，引擎只需调用 wrap/wrap_all 即可。

    注：Engine 本身现在也支持自动 blob offload（见 Engine(blob_store=..., default_max_result_chars=...)），
    ToolExecutor 仍保留用于"外部按名字定制 cap"的场景；新代码建议走 Engine 路径。
    """
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
        """返回新的 ToolSpec 而非修改原始对象，保持不可变语义。
        通过 dataclasses.replace 复制所有字段，避免新增字段（trust_level / read_only
        等）被悄悄丢弃——历史教训：之前硬构造 ToolSpec 把 trust_level 搞没了。"""
        cap = self._caps.get(spec.name, self._default_cap)
        blob_store = self._blob_store
        original_handler = spec.handler

        async def capped_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
            output = await original_handler(args, ctx)
            result = enforce_cap(output, cap, blob_store)
            return result.output

        return replace(spec, handler=capped_handler)

    def wrap_all(self, specs: list[ToolSpec]) -> list[ToolSpec]:
        return [self.wrap(spec) for spec in specs]
