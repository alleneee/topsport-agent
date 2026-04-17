from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .blob_store import BlobStore

# 按工具类型设定字符上限，防止单次工具输出撑爆 LLM 上下文窗口。
DEFAULT_CAPS: dict[str, int] = {
    "read_file": 20_000,
    "search": 10_000,
    "shell": 15_000,
}
DEFAULT_CAP = 20_000


def serialize_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str, ensure_ascii=False)
    except Exception:
        return str(output)


@dataclass
class CapResult:
    output: Any
    truncated: bool = False
    original_size: int = 0
    blob_ref: str | None = None


def enforce_cap(
    output: Any,
    cap: int,
    blob_store: BlobStore | None = None,
) -> CapResult:
    """超限时：有 blob_store 则全量落盘并返回 blob_ref + preview；无则只返回截断预览。"""
    serialized = serialize_output(output)
    size = len(serialized)

    if size <= cap:
        return CapResult(output=output, original_size=size)

    if blob_store is not None:
        blob_ref = blob_store.store(serialized)
        return CapResult(
            output={
                "truncated": True,
                "original_size": size,
                "cap": cap,
                "blob_ref": blob_ref,
                "preview": serialized[:cap],
            },
            truncated=True,
            original_size=size,
            blob_ref=blob_ref,
        )

    return CapResult(
        output={
            "truncated": True,
            "original_size": size,
            "cap": cap,
            "preview": serialized[:cap],
        },
        truncated=True,
        original_size=size,
    )
