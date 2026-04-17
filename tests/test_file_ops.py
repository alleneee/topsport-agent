"""文件操作工具测试：read/write/edit/list_dir/glob/grep。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from topsport_agent.tools.file_ops import file_tools
from topsport_agent.types.tool import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="s", call_id="c", cancel_event=asyncio.Event())


def _find(name: str):
    for t in file_tools():
        if t.name == name:
            return t
    raise KeyError(name)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


async def test_read_file_basic(tmp_path: Path) -> None:
    p = tmp_path / "hello.txt"
    p.write_text("line1\nline2\nline3\n")
    result = await _find("read_file").handler({"path": str(p)}, _ctx())
    assert result["ok"] is True
    assert "1\tline1" in result["content"]
    assert "3\tline3" in result["content"]
    assert result["total_lines"] == 3


async def test_read_file_with_offset_limit(tmp_path: Path) -> None:
    p = tmp_path / "big.txt"
    p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
    result = await _find("read_file").handler(
        {"path": str(p), "offset": 3, "limit": 2}, _ctx()
    )
    assert result["ok"] is True
    assert "3\tline3" in result["content"]
    assert "4\tline4" in result["content"]
    assert "5\tline5" not in result["content"]
    assert result["returned_lines"] == 2


async def test_read_file_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        await _find("read_file").handler({"path": "relative/path.txt"}, _ctx())


async def test_read_file_not_found(tmp_path: Path) -> None:
    result = await _find("read_file").handler(
        {"path": str(tmp_path / "ghost.txt")}, _ctx()
    )
    assert result["ok"] is False
    assert "not found" in result["error"]


async def test_read_file_rejects_directory(tmp_path: Path) -> None:
    result = await _find("read_file").handler({"path": str(tmp_path)}, _ctx())
    assert result["ok"] is False
    assert "not a file" in result["error"]


async def test_read_file_truncates_long_line(tmp_path: Path) -> None:
    p = tmp_path / "long.txt"
    p.write_text("x" * 5000)
    result = await _find("read_file").handler({"path": str(p)}, _ctx())
    assert result["ok"] is True
    assert "truncated" in result["content"]


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


async def test_write_file_creates_new(tmp_path: Path) -> None:
    p = tmp_path / "new.txt"
    result = await _find("write_file").handler(
        {"path": str(p), "content": "hello"}, _ctx()
    )
    assert result["ok"] is True
    assert result["created"] is True
    assert p.read_text() == "hello"


async def test_write_file_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "exists.txt"
    p.write_text("old")
    result = await _find("write_file").handler(
        {"path": str(p), "content": "new"}, _ctx()
    )
    assert result["ok"] is True
    assert result["created"] is False
    assert p.read_text() == "new"


async def test_write_file_rejects_missing_parent(tmp_path: Path) -> None:
    p = tmp_path / "nonexistent-dir" / "file.txt"
    result = await _find("write_file").handler(
        {"path": str(p), "content": "x"}, _ctx()
    )
    assert result["ok"] is False
    assert "parent directory" in result["error"]


async def test_write_file_rejects_non_string_content(tmp_path: Path) -> None:
    result = await _find("write_file").handler(
        {"path": str(tmp_path / "x.txt"), "content": 123}, _ctx()
    )
    assert result["ok"] is False
    assert "must be a string" in result["error"]


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


async def test_edit_file_single_replacement(tmp_path: Path) -> None:
    p = tmp_path / "code.py"
    p.write_text("def foo():\n    return 1\n")
    result = await _find("edit_file").handler(
        {"path": str(p), "old_string": "return 1", "new_string": "return 42"}, _ctx()
    )
    assert result["ok"] is True
    assert result["replacements"] == 1
    assert "return 42" in p.read_text()


async def test_edit_file_rejects_non_unique_match(tmp_path: Path) -> None:
    p = tmp_path / "dup.txt"
    p.write_text("foo\nfoo\nfoo\n")
    result = await _find("edit_file").handler(
        {"path": str(p), "old_string": "foo", "new_string": "bar"}, _ctx()
    )
    assert result["ok"] is False
    assert result["match_count"] == 3


async def test_edit_file_replace_all(tmp_path: Path) -> None:
    p = tmp_path / "dup.txt"
    p.write_text("foo\nfoo\nfoo\n")
    result = await _find("edit_file").handler(
        {"path": str(p), "old_string": "foo", "new_string": "bar", "replace_all": True}, _ctx()
    )
    assert result["ok"] is True
    assert result["replacements"] == 3
    assert p.read_text() == "bar\nbar\nbar\n"


async def test_edit_file_rejects_missing_match(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("actual content")
    result = await _find("edit_file").handler(
        {"path": str(p), "old_string": "absent", "new_string": "present"}, _ctx()
    )
    assert result["ok"] is False
    assert result["match_count"] == 0


async def test_edit_file_rejects_identical_strings(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("content")
    result = await _find("edit_file").handler(
        {"path": str(p), "old_string": "x", "new_string": "x"}, _ctx()
    )
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


async def test_list_dir(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    result = await _find("list_dir").handler({"path": str(tmp_path)}, _ctx())
    assert result["ok"] is True
    names = {e["name"] for e in result["entries"]}
    assert names == {"a.txt", "sub"}
    types = {e["name"]: e["type"] for e in result["entries"]}
    assert types["a.txt"] == "file"
    assert types["sub"] == "dir"


async def test_list_dir_rejects_file(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("")
    result = await _find("list_dir").handler({"path": str(p)}, _ctx())
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# glob_files
# ---------------------------------------------------------------------------


async def test_glob_files_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    result = await _find("glob_files").handler(
        {"pattern": "**/*.py", "path": str(tmp_path)}, _ctx()
    )
    assert result["ok"] is True
    assert result["count"] == 2
    assert any("a.py" in m for m in result["matches"])
    assert any("b.py" in m for m in result["matches"])


async def test_glob_files_no_match(tmp_path: Path) -> None:
    result = await _find("glob_files").handler(
        {"pattern": "*.nonexistent", "path": str(tmp_path)}, _ctx()
    )
    assert result["ok"] is True
    assert result["count"] == 0


# ---------------------------------------------------------------------------
# grep_files
# ---------------------------------------------------------------------------


async def test_grep_files_basic(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("class Bar:\n    pass\n")
    result = await _find("grep_files").handler(
        {"pattern": r"def \w+", "path": str(tmp_path)}, _ctx()
    )
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["matches"][0]["line_number"] == 1


async def test_grep_files_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("target\n")
    (tmp_path / "b.txt").write_text("target\n")
    result = await _find("grep_files").handler(
        {"pattern": "target", "path": str(tmp_path), "glob": "*.py"}, _ctx()
    )
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["matches"][0]["file"].endswith("a.py")


async def test_grep_files_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Hello World\n")
    result = await _find("grep_files").handler(
        {"pattern": "HELLO", "path": str(tmp_path), "case_insensitive": True}, _ctx()
    )
    assert result["ok"] is True
    assert result["count"] == 1


async def test_grep_files_invalid_regex(tmp_path: Path) -> None:
    result = await _find("grep_files").handler(
        {"pattern": "[invalid", "path": str(tmp_path)}, _ctx()
    )
    assert result["ok"] is False
    assert "invalid regex" in result["error"]


# ---------------------------------------------------------------------------
# file_tools() exposes all tools
# ---------------------------------------------------------------------------


def test_file_tools_returns_all() -> None:
    names = {t.name for t in file_tools()}
    assert names == {
        "read_file", "write_file", "edit_file",
        "list_dir", "glob_files", "grep_files",
    }


def test_file_tools_have_schemas() -> None:
    for tool in file_tools():
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters
        assert tool.description
