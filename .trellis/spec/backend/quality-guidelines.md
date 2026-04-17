# Quality Guidelines

> Code quality standards for the agent runtime.

---

## Overview

Python 3.11+. Linted with ruff (E/F/I/UP, line-length 120). Tested with
pytest-asyncio (auto mode). 143+ tests, all must pass before any work is
reported as complete.

---

## Forbidden Patterns

| Pattern | Why | Alternative |
|---------|-----|-------------|
| `subprocess.Popen(cmd, shell=True)` | Shell injection | `tools/safe_shell.safe_exec(["cmd", "arg"])` |
| `from anthropic import X` at module level | Breaks when optional dep not installed | `importlib.import_module(variable)` inside function |
| `except Exception: pass` | Silent failures | `except Exception: _logger.debug(exc_info=True)` |
| `session.messages.append(...)` inside ContextProvider | Ephemeral context must not persist | Return messages from `provide()`, engine handles merge |
| Inserting messages between `tool_calls` and `tool_result` | API 400 from Anthropic and OpenAI | Use InterjectQueue to defer until after tool results |
| `asyncio.Event()` reassignment for cancel | Stale references in ToolContext | `self._cancel_event.clear()` to reuse same object |
| Path join with unsanitized user input | Path traversal | Validate with regex or `path.resolve().is_relative_to(base)` |

---

## Required Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| Protocol-based hooks for extensibility | `engine/hooks.py` | Engine stays decoupled from memory/skills/mcp/observability |
| `client=` constructor injection | All optional-dep modules | Tests run without installing optional packages |
| `_key_to_filename(key)` via sha256 hash | `memory/file_store.py` | Prevents key collision on filesystem |
| `OrderedDict` with `max_sessions` | `LoopDetector`, `InterjectQueue` | Prevents unbounded memory growth |
| `asyncio.wait(FIRST_COMPLETED)` for cancel | `engine/loop.py` | Preempts LLM calls without polling |

---

## Testing Requirements

- All tests run via `uv run pytest -v`
- Tests must not depend on optional packages (anthropic, openai, langfuse, mcp)
- Use mock clients and injectable factories for all external dependencies
- Every architectural invariant has at least one dedicated test
- Run full suite after every code change, no exceptions

---

## Code Review Checklist

Verified by multi-model review (Claude Security + Maintainability + Adversarial + Codex GPT-5.4):

1. Path traversal: all user-controlled path components validated
2. Cancel propagation: cancel event reaches all async boundaries
3. Resource cleanup: no unbounded dicts without eviction
4. Error visibility: no silent `except: pass` blocks
5. Protocol compliance: tool_calls immediately followed by tool_results
6. Optional dep isolation: no top-level imports of optional packages
7. Output caps: tool output truncated before entering session messages
