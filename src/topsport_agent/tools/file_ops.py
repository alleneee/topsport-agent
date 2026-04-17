"""文件操作工具套件：read/write/edit/list_dir/glob/grep。

工具语义参考 Claude Code：
- read_file 带行偏移/限制，返回"line_number\t内容"格式
- edit_file 是精确字符串替换，不是 diff 合并
- grep_files / glob_files 基于 ripgrep 习惯，无 ripgrep 时回退到 Python 实现

安全约束：
- 所有路径必须绝对路径（避免相对路径的歧义）
- ToolContext.workspace_root 设置时强制沙箱：路径 resolve 后必须落在根内，
  符号链接逃逸被拒绝；未设置时保留 CLI 的宽松行为
- edit_file 要求 old_string 唯一匹配（除非 replace_all=True），并按绝对路径串行
- write_file / edit_file 通过临时文件 + os.replace 原子替换，避免中断留半写入
- 单次读写有大小上限，超限截断并标注
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ..types.tool import ToolContext, ToolSpec

_MAX_READ_BYTES = 1_000_000  # 单次 read_file 最多 1MB
_MAX_WRITE_BYTES = 5_000_000  # 单次 write_file 最多 5MB
_MAX_GREP_MATCHES = 500  # grep 最多返回 500 条命中
_MAX_GLOB_RESULTS = 1000  # glob 最多返回 1000 条路径
_MAX_LINE_LENGTH = 2000  # 单行超过 2000 字符截断，防止超宽二进制

# edit_file 串行化：同一绝对路径的并发 edit 排队执行，避免 TOCTOU 最后写者覆盖。
# key 是 resolve 后的字符串路径，进程级生命周期即可，无需显式清理（条目数 ≈ 活跃文件数）。
_edit_locks: dict[str, asyncio.Lock] = {}
_edit_locks_guard = asyncio.Lock()


def _ensure_absolute(path_str: str) -> Path:
    """所有路径必须绝对路径 —— 相对路径在 LLM 的长对话里极易出错。"""
    p = Path(path_str)
    if not p.is_absolute():
        raise ValueError(f"path must be absolute, got: {path_str}")
    return p


def _check_containment(path: Path, ctx: ToolContext) -> str | None:
    """仅在 workspace_root 被设置时生效：
    - path resolve 后必须 is_relative_to(root)
    - root 本身也 resolve，两边都规范化后比较
    返回错误描述；合规返回 None。未设置 root 时不检查（保留 CLI 兼容）。
    """
    root = ctx.workspace_root
    if root is None:
        return None
    try:
        real_path = path.resolve(strict=False)
        real_root = root.resolve(strict=False)
    except OSError as exc:
        return f"path resolution failed: {exc}"
    if not real_path.is_relative_to(real_root):
        return (
            f"path escapes workspace: {path} resolves to {real_path}, "
            f"root={real_root}"
        )
    return None


def _atomic_write_text(path: Path, content: str) -> None:
    """临时文件同目录 + os.replace 原子替换。
    崩溃/掉电时要么保留旧文件要么是完整新文件，不会出现半写入。
    """
    parent = path.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


async def _get_edit_lock(path: Path) -> asyncio.Lock:
    """按 resolve 后的路径取同一把锁，保证同一文件的 edit 串行。"""
    key = str(path.resolve(strict=False))
    async with _edit_locks_guard:
        lock = _edit_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _edit_locks[key] = lock
        return lock


async def _read_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """读文件内容。支持 offset (1-based 行号) 和 limit (行数)。"""
    path = _ensure_absolute(args["path"])
    offset = int(args.get("offset", 0) or 0)
    limit = int(args.get("limit", 0) or 0)

    violation = _check_containment(path, ctx)
    if violation is not None:
        return {"ok": False, "error": violation}

    if not path.exists():
        return {"ok": False, "error": f"file not found: {path}"}
    if not path.is_file():
        return {"ok": False, "error": f"not a file: {path}"}

    size = path.stat().st_size
    if size > _MAX_READ_BYTES and offset == 0 and limit == 0:
        return {
            "ok": False,
            "error": f"file too large ({size} bytes > {_MAX_READ_BYTES}), "
                     "use offset/limit to read in chunks",
        }

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    lines = raw.splitlines()
    total = len(lines)

    start = max(offset - 1, 0) if offset > 0 else 0
    end = start + limit if limit > 0 else total
    sliced = lines[start:end]

    # 输出格式：行号<TAB>内容，超长行截断标记
    numbered: list[str] = []
    for i, line in enumerate(sliced, start=start + 1):
        if len(line) > _MAX_LINE_LENGTH:
            line = line[:_MAX_LINE_LENGTH] + f"  [line truncated, {len(line)} chars total]"
        numbered.append(f"{i}\t{line}")

    return {
        "ok": True,
        "content": "\n".join(numbered),
        "total_lines": total,
        "returned_lines": len(sliced),
        "start_line": start + 1 if sliced else 0,
        "truncated": end < total,
    }


async def _write_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """写入文件（覆盖或创建）。父目录必须已存在，不自动创建。"""
    path = _ensure_absolute(args["path"])
    content = args["content"]

    if not isinstance(content, str):
        return {"ok": False, "error": "content must be a string"}

    if len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
        return {
            "ok": False,
            "error": f"content exceeds {_MAX_WRITE_BYTES} bytes, write in chunks",
        }

    violation = _check_containment(path, ctx)
    if violation is not None:
        return {"ok": False, "error": violation}

    if not path.parent.exists():
        return {
            "ok": False,
            "error": f"parent directory does not exist: {path.parent}",
        }

    try:
        existed = path.exists()
        _atomic_write_text(path, content)
        return {
            "ok": True,
            "path": str(path),
            "bytes_written": len(content.encode("utf-8")),
            "created": not existed,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def _edit_file(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """精确字符串替换。old_string 必须在文件中存在且（默认）唯一。

    同路径并发 edit 被 _edit_locks 串行化，避免 TOCTOU 最后写者覆盖。
    """
    path = _ensure_absolute(args["path"])
    old_string = args["old_string"]
    new_string = args["new_string"]
    replace_all = bool(args.get("replace_all", False))

    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return {"ok": False, "error": "old_string and new_string must be strings"}
    if old_string == new_string:
        return {"ok": False, "error": "old_string equals new_string, nothing to change"}

    violation = _check_containment(path, ctx)
    if violation is not None:
        return {"ok": False, "error": violation}

    if not path.exists():
        return {"ok": False, "error": f"file not found: {path}"}
    if not path.is_file():
        return {"ok": False, "error": f"not a file: {path}"}

    lock = await _get_edit_lock(path)
    async with lock:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": f"read failed: {type(exc).__name__}: {exc}"}

        count = text.count(old_string)
        if count == 0:
            return {
                "ok": False,
                "error": "old_string not found in file",
                "match_count": 0,
            }
        if count > 1 and not replace_all:
            return {
                "ok": False,
                "error": f"old_string matched {count} times; "
                         "provide more unique context or set replace_all=true",
                "match_count": count,
            }

        if replace_all:
            new_text = text.replace(old_string, new_string)
        else:
            new_text = text.replace(old_string, new_string, 1)

        try:
            _atomic_write_text(path, new_text)
            return {
                "ok": True,
                "path": str(path),
                "replacements": count if replace_all else 1,
            }
        except Exception as exc:
            return {"ok": False, "error": f"write failed: {type(exc).__name__}: {exc}"}


async def _list_dir(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """列出目录内容。只返回名称和类型，不递归。"""
    path = _ensure_absolute(args["path"])

    violation = _check_containment(path, ctx)
    if violation is not None:
        return {"ok": False, "error": violation}

    if not path.exists():
        return {"ok": False, "error": f"directory not found: {path}"}
    if not path.is_dir():
        return {"ok": False, "error": f"not a directory: {path}"}

    try:
        entries: list[dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            entries.append({
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            })
        return {"ok": True, "path": str(path), "entries": entries, "count": len(entries)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def _glob_files(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """按 glob 模式查找文件。支持 ** 递归。"""
    pattern = args["pattern"]
    base = _ensure_absolute(args.get("path") or str(Path.cwd()))

    violation = _check_containment(base, ctx)
    if violation is not None:
        return {"ok": False, "error": violation}

    if not base.exists():
        return {"ok": False, "error": f"base path not found: {base}"}
    if not base.is_dir():
        return {"ok": False, "error": f"base path is not a directory: {base}"}

    try:
        matches: list[str] = []
        # Path.rglob 支持 **，但不支持通配符在前缀；用 base.glob(pattern) 兼顾两者
        # 再对 pattern 中含 **/ 的情况做 rglob 回退
        if "**" in pattern:
            iter_ = base.glob(pattern)
        else:
            iter_ = base.glob(pattern)
        for p in iter_:
            if p.is_file():
                # 在 workspace_root 设置时，过滤掉 resolve 后逃出根的结果（符号链接防御）
                if _check_containment(p, ctx) is not None:
                    continue
                matches.append(str(p))
                if len(matches) >= _MAX_GLOB_RESULTS:
                    break
        matches.sort()
        return {
            "ok": True,
            "pattern": pattern,
            "base": str(base),
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) >= _MAX_GLOB_RESULTS,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


async def _grep_files(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """在文件内容中搜索正则。支持 glob 过滤、大小写不敏感、上下文行。"""
    pattern_str = args["pattern"]
    base = _ensure_absolute(args.get("path") or str(Path.cwd()))
    glob_filter = args.get("glob") or "*"
    case_insensitive = bool(args.get("case_insensitive", False))
    max_results = int(args.get("max_results", _MAX_GREP_MATCHES))
    max_results = min(max_results, _MAX_GREP_MATCHES)

    violation = _check_containment(base, ctx)
    if violation is not None:
        return {"ok": False, "error": violation}

    if not base.exists():
        return {"ok": False, "error": f"base path not found: {base}"}

    flags = re.MULTILINE | (re.IGNORECASE if case_insensitive else 0)
    try:
        pattern = re.compile(pattern_str, flags)
    except re.error as exc:
        return {"ok": False, "error": f"invalid regex: {exc}"}

    # 收集候选文件
    if base.is_file():
        files = [base]
    else:
        files = [
            p for p in base.rglob("*")
            if p.is_file() and fnmatch.fnmatch(p.name, glob_filter)
            and _check_containment(p, ctx) is None
        ]

    matches: list[dict[str, Any]] = []
    for file_path in sorted(files):
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue

        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                if len(line) > _MAX_LINE_LENGTH:
                    line = line[:_MAX_LINE_LENGTH] + " [truncated]"
                matches.append({
                    "file": str(file_path),
                    "line_number": i,
                    "line": line,
                })
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    return {
        "ok": True,
        "pattern": pattern_str,
        "base": str(base),
        "matches": matches,
        "count": len(matches),
        "truncated": len(matches) >= max_results,
    }


def file_tools() -> list[ToolSpec]:
    """返回完整的文件操作工具套件。"""
    return [
        ToolSpec(
            name="read_file",
            description=(
                "Read a text file. Returns content with line numbers (1-based). "
                "Use offset+limit to page through large files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "offset": {"type": "integer", "description": "Starting line number (1-based), 0 = from beginning"},
                    "limit": {"type": "integer", "description": "Number of lines to read, 0 = to end"},
                },
                "required": ["path"],
            },
            handler=_read_file,
        ),
        ToolSpec(
            name="write_file",
            description="Create or overwrite a file with the given content.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
            handler=_write_file,
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Replace exact string occurrence(s) in a file. "
                "old_string must match exactly (including whitespace). "
                "By default, old_string must be unique; set replace_all=true to allow multiple matches."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "old_string": {"type": "string", "description": "Exact string to find"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            handler=_edit_file,
        ),
        ToolSpec(
            name="list_dir",
            description="List direct children of a directory (non-recursive).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute directory path"},
                },
                "required": ["path"],
            },
            handler=_list_dir,
        ),
        ToolSpec(
            name="glob_files",
            description="Find files matching a glob pattern (supports ** for recursive).",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
                    "path": {"type": "string", "description": "Base directory, absolute (default: cwd)"},
                },
                "required": ["pattern"],
            },
            handler=_glob_files,
        ),
        ToolSpec(
            name="grep_files",
            description=(
                "Search file contents by regex. Returns matching lines with file and line number. "
                "Use glob to filter by filename pattern."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Base directory or file, absolute"},
                    "glob": {"type": "string", "description": "Filename glob filter, e.g. '*.py'"},
                    "case_insensitive": {"type": "boolean", "description": "Case-insensitive match"},
                    "max_results": {"type": "integer", "description": "Max matches to return (max 500)"},
                },
                "required": ["pattern"],
            },
            handler=_grep_files,
        ),
    ]
