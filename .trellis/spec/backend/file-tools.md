# File Operation Tools

> Contracts for read_file / write_file / edit_file / list_dir / glob_files / grep_files.

---

## Overview

Six file tools in `src/topsport_agent/tools/file_ops.py`, exported via
`tools.file_tools() -> list[ToolSpec]`. Default-agent auto-mounts them
(`enable_file_ops=True`).

All tools share:
- **Absolute paths only.** Relative paths raise `ValueError` BEFORE any I/O.
- **Return dict with `ok: bool`** as the first field.
- **No raise on expected failures** (file missing, no match, invalid regex) —
  failures become `{ok: False, error: ...}` results.

---

## Size and Result Limits

Defined at module level:

| Constant | Value | Applies to |
| --- | --- | --- |
| `_MAX_READ_BYTES` | 1_000_000 (1MB) | `read_file` without offset/limit |
| `_MAX_WRITE_BYTES` | 5_000_000 (5MB) | `write_file` content size |
| `_MAX_GREP_MATCHES` | 500 | `grep_files` max hits |
| `_MAX_GLOB_RESULTS` | 1000 | `glob_files` max paths |
| `_MAX_LINE_LENGTH` | 2000 chars | Per-line truncation in read/grep |

Exceeding a limit returns `{ok: False, error: ...}` OR `{ok: True, truncated: True}`
depending on the tool. See per-tool section.

---

## Tool Contracts

### read_file

```json
{
  "path": "<absolute path>",
  "offset": <int, 1-based line, 0 means from start>,
  "limit": <int, 0 means to end>
}
```

**Success return:**
```python
{
  "ok": True,
  "content": "<line_number>\t<line text>\n...",   # tab-separated, 1-based line numbers
  "total_lines": <int>,
  "returned_lines": <int>,
  "start_line": <int>,
  "truncated": <bool>,
}
```

**Behavior:**
- File > `_MAX_READ_BYTES` and offset=0 and limit=0 → error "file too large"
- Individual lines > `_MAX_LINE_LENGTH` are truncated inline with marker
- Non-existent path → `{ok: False, error: "file not found"}`
- Path is directory → `{ok: False, error: "not a file"}`

### write_file

```json
{ "path": "<absolute>", "content": "<string>" }
```

**Success:** `{ok: True, path, bytes_written, created: <True if newly created>}`

**Failures:**
- `content` not a str → `{ok: False, error: "content must be a string"}`
- bytes > `_MAX_WRITE_BYTES` → `{ok: False, error: "content exceeds ..."}`
- Parent dir missing → `{ok: False, error: "parent directory does not exist"}`
- DOES NOT auto-create parent directories (caller must ensure).
- Always overwrites existing file (no confirmation).

### edit_file

```json
{
  "path": "<absolute>",
  "old_string": "<exact text>",
  "new_string": "<replacement>",
  "replace_all": <bool, default false>
}
```

**Uniqueness rule (critical invariant):**
- Default (`replace_all=false`): `old_string` MUST match EXACTLY once in file.
- `replace_all=true`: all occurrences replaced.
- Zero matches → `{ok: False, error: "old_string not found", match_count: 0}`
- >1 matches, replace_all=false → `{ok: False, error: "matched N times", match_count: N}`
- `old_string == new_string` → `{ok: False, error: "nothing to change"}`

**Success:** `{ok: True, path, replacements: <count>}`

### list_dir

```json
{ "path": "<absolute dir>" }
```

**Success:**
```python
{
  "ok": True, "path": ..., "count": <int>,
  "entries": [
    {"name": str, "type": "file"|"dir", "size": int|None}, ...
  ]
}
```

- Non-recursive, one level only.
- Sorted by name (`Path.iterdir()` + `sorted()`).
- `size` is `None` for directories.

### glob_files

```json
{ "pattern": "<glob>", "path": "<optional absolute base, default cwd>" }
```

**Success:**
```python
{
  "ok": True, "pattern", "base", "count",
  "matches": [<absolute path>, ...],
  "truncated": <bool>,
}
```

- Supports `**` for recursive.
- Returns ONLY files (not directories).
- Truncates at `_MAX_GLOB_RESULTS`.

### grep_files

```json
{
  "pattern": "<regex>",
  "path": "<optional absolute, file or dir>",
  "glob": "<filename glob filter, default '*'>",
  "case_insensitive": <bool, default false>,
  "max_results": <int, capped at 500>
}
```

**Success:**
```python
{
  "ok": True, "pattern", "base", "count",
  "matches": [
    {"file": <abs path>, "line_number": <int>, "line": <str>}, ...
  ],
  "truncated": <bool>,
}
```

- Regex uses `re.MULTILINE` + optional `re.IGNORECASE`.
- Invalid regex → `{ok: False, error: "invalid regex: ..."}`.
- Lines > `_MAX_LINE_LENGTH` are truncated inline.
- Files unreadable as UTF-8 (replace errors) are silently skipped.

---

## Integration

Default agent auto-mounts via `extra_tools`:

```python
# src/topsport_agent/agent/default.py
def default_agent(..., enable_file_ops: bool = True, ...):
    tools = list(extra_tools or [])
    if enable_file_ops:
        tools = file_tools() + tools   # prepend so user extras don't clash
    ...
```

Ordering: `file_tools()` are placed BEFORE user-provided `extra_tools`. This
means user tools with colliding names win (Engine deduplicates by first-seen).

---

## Forbidden Patterns

- DO NOT accept relative paths. Always use `_ensure_absolute(path_str)` helper.
- DO NOT auto-create parent directories on `write_file` (explicit `mkdir` by caller).
- DO NOT raise on "expected" failures (file not found, no match). Return `{ok: False}`.
- DO NOT skip the single-match requirement in `edit_file` without `replace_all`.
  It exists to prevent LLM from accidentally editing unintended occurrences.

---

## Tests

Reference tests in `tests/test_file_ops.py` (25 tests):
- Full CRUD coverage + edge cases (empty, missing, too large, too long lines)
- Absolute-path enforcement
- `edit_file` single-match and replace_all semantics
- `grep_files` glob filter + case sensitivity + regex error
- `file_tools()` exposure completeness
