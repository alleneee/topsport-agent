from __future__ import annotations

from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .store import MemoryStore
from .types import MemoryEntry, MemoryType


def build_memory_tools(store: MemoryStore) -> list[ToolSpec]:
    """闭包工厂：通过 store 参数注入具体存储实现，返回的 ToolSpec 列表可直接交给 Engine。"""
    async def save_memory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        raw_type = args.get("type", MemoryType.NOTE.value)
        # 类型校验失败时返回合法枚举值列表，引导 LLM 自行修正参数
        try:
            memory_type = MemoryType(raw_type)
        except ValueError:
            return {
                "ok": False,
                "error": f"invalid type '{raw_type}'",
                "allowed": [t.value for t in MemoryType],
            }
        entry = MemoryEntry(
            key=args["key"],
            name=args.get("name") or args["key"],
            description=args.get("description", ""),
            type=memory_type,
            content=args["content"],
        )
        await store.write(ctx.session_id, entry)
        return {"ok": True, "key": entry.key, "type": entry.type.value}

    async def recall_memory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        """recall 双模式：传 key 精确读取单条，省略 key 列出全部摘要（不含 content），控制返回体积。"""
        key = args.get("key")
        if key:
            entry = await store.read(ctx.session_id, key)
            if entry is None:
                return {"found": False, "key": key}
            return {
                "found": True,
                "entry": {
                    "key": entry.key,
                    "name": entry.name,
                    "description": entry.description,
                    "type": entry.type.value,
                    "content": entry.content,
                },
            }
        entries = await store.list(ctx.session_id)
        return {
            "count": len(entries),
            "entries": [
                {
                    "key": entry.key,
                    "name": entry.name,
                    "type": entry.type.value,
                }
                for entry in entries
            ],
        }

    async def forget_memory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        deleted = await store.delete(ctx.session_id, args["key"])
        return {"deleted": deleted, "key": args["key"]}

    save_spec = ToolSpec(
        name="save_memory",
        description="Persist a working-memory entry for the current session.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": [t.value for t in MemoryType],
                },
                "content": {"type": "string"},
            },
            "required": ["key", "content"],
        },
        handler=save_memory,
    )

    recall_spec = ToolSpec(
        name="recall_memory",
        description="Read working-memory entries. Omit 'key' to list all entries.",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
        },
        handler=recall_memory,
    )

    forget_spec = ToolSpec(
        name="forget_memory",
        description="Delete a working-memory entry by key.",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        handler=forget_memory,
    )

    return [save_spec, recall_spec, forget_spec]
