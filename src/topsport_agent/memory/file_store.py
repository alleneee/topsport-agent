from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime
from pathlib import Path

from .types import MemoryEntry, MemoryType

# 路径组件白名单：阻止 session_id 携带 ../ 或特殊字符实现路径穿越
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def _validate_path_component(value: str, label: str) -> str:
    if not value or not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"{label} must be non-empty and contain only [a-zA-Z0-9._-], got: {value!r}"
        )
    return value


def _key_to_filename(key: str) -> str:
    """key 可以是任意用户输入，SHA-256 截断映射为安全文件名，同时保持单射（24 hex 足够避免碰撞）。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _format_frontmatter(entry: MemoryEntry) -> str:
    """持久化格式：YAML frontmatter + Markdown body，人类可直接编辑，git 友好。"""
    lines = [
        "---",
        f"name: {entry.name}",
        f"description: {entry.description}",
        f"type: {entry.type.value}",
        f"key: {entry.key}",
    ]
    if entry.created_at:
        lines.append(f"created_at: {entry.created_at.isoformat()}")
    if entry.updated_at:
        lines.append(f"updated_at: {entry.updated_at.isoformat()}")
    lines.append("---")
    lines.append("")
    lines.append(entry.content)
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    header = text[4:end]
    body = text[end + 5 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class FileMemoryStore:
    """文件系统实现：每条记忆一个 .md 文件，按 session_id 子目录隔离。

    所有阻塞 IO 通过 asyncio.to_thread 卸载到线程池，保护事件循环不被磁盘延迟拖慢。
    """
    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def _session_dir(self, session_id: str) -> Path:
        _validate_path_component(session_id, "session_id")
        return self._base_path / session_id

    def _file_path(self, session_id: str, key: str) -> Path:
        return self._session_dir(session_id) / f"{_key_to_filename(key)}.md"

    async def write(self, session_id: str, entry: MemoryEntry) -> None:
        now = datetime.now()
        if entry.created_at is None:
            entry.created_at = now
        entry.updated_at = now
        session_dir = self._session_dir(session_id)
        path = self._file_path(session_id, entry.key)
        payload = _format_frontmatter(entry)
        # 文件 IO 放到线程池里，避免阻塞事件循环
        await asyncio.to_thread(self._write_file, path, session_dir, payload)

    @staticmethod
    def _write_file(path: Path, session_dir: Path, payload: str) -> None:
        session_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    async def read(self, session_id: str, key: str) -> MemoryEntry | None:
        path = self._file_path(session_id, key)
        content = await asyncio.to_thread(self._read_file, path)
        if content is None:
            return None
        return self._entry_from_text(key, content)

    @staticmethod
    def _read_file(path: Path) -> str | None:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _entry_from_text(fallback_key: str, text: str) -> MemoryEntry:
        meta, body = _parse_frontmatter(text)
        type_value = meta.get("type", MemoryType.NOTE.value)
        try:
            memory_type = MemoryType(type_value)
        except ValueError:
            memory_type = MemoryType.NOTE
        # 读取时优先信任 frontmatter，缺字段才回退到文件名或默认值
        return MemoryEntry(
            key=meta.get("key", fallback_key),
            name=meta.get("name", fallback_key),
            description=meta.get("description", ""),
            type=memory_type,
            content=body.rstrip("\n"),
            created_at=_parse_datetime(meta.get("created_at")),
            updated_at=_parse_datetime(meta.get("updated_at")),
        )

    async def list(self, session_id: str) -> list[MemoryEntry]:
        session_dir = self._session_dir(session_id)
        files = await asyncio.to_thread(self._list_files, session_dir)
        entries: list[MemoryEntry] = []
        for path in files:
            text = await asyncio.to_thread(self._read_file, path)
            if text is None:
                continue
            entries.append(self._entry_from_text(path.stem, text))
        # 列表接口返回稳定顺序，避免调用方每次自己排序
        entries.sort(key=lambda e: e.key)
        return entries

    @staticmethod
    def _list_files(session_dir: Path) -> list[Path]:
        if not session_dir.exists():
            return []
        return sorted(session_dir.glob("*.md"))

    async def delete(self, session_id: str, key: str) -> bool:
        path = self._file_path(session_id, key)
        return await asyncio.to_thread(self._delete_file, path)

    @staticmethod
    def _delete_file(path: Path) -> bool:
        if not path.exists():
            return False
        path.unlink()
        return True
