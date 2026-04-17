# Database Guidelines

> Storage patterns in this project.

---

## Overview

This project has no traditional database. All persistence uses file-based storage
or in-memory data structures with Protocol-based interfaces for future swapping.

---

## Storage Backends

| What | Backend | Location | Interface |
|------|---------|----------|-----------|
| Working memory | Markdown files with frontmatter | `<base>/<session_id>/<hash>.md` | `MemoryStore` Protocol |
| Tool output blobs | Plain text files | `<base>/<sha256>.blob` | `BlobStore` Protocol |
| Skills | SKILL.md files (read-only) | `~/.claude/skills/<name>/SKILL.md` | `SkillRegistry` |
| MCP config | JSON file (read-only) | User-specified path | `load_mcp_config()` |
| Session state | In-memory `Session` dataclass | N/A | Direct attribute access |
| Langfuse traces | External (Langfuse API) | N/A | `LangfuseTracer` EventSubscriber |

---

## Future Database Path

If a real database is needed (multi-process, persistence across restarts),
the migration path is:

1. Implement `MemoryStore` Protocol with SQLite/Postgres backend
2. Implement `BlobStore` Protocol with S3-compatible backend
3. Add session persistence (serialize `Session` to DB on checkpoint)

The Protocol interfaces are already defined. No engine changes needed.
