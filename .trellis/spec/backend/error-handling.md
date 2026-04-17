# Error Handling

> Three-layer error model used across the runtime.

---

## Overview

The project uses a deliberate three-layer error model. Each layer has its own
convention. Mixing them is a bug.

---

## Error Types

| Exception | Module | When |
|-----------|--------|------|
| `Cancelled` | `engine/loop.py` | Engine cancel signal detected |
| `ShellInjectionError` | `tools/safe_shell.py` | String command or shell interpreter with `-c` |
| `ValueError` | `memory/file_store.py` | Invalid `session_id` format |
| `ImportError` | adapters, langfuse, mcp | Optional dependency not installed |
| `RuntimeError` | `engine/concurrency.py` | Session already running (EngineGuard) |

---

## Error Handling Patterns

### Layer 1: Engine protocol (exceptions)

Errors that break the ReAct loop use exceptions. Engine catches them and emits
`EventType.ERROR` or `EventType.CANCELLED`.

```python
except Cancelled:
    yield self._event(EventType.CANCELLED, session, {})
except Exception as exc:
    yield self._event(EventType.ERROR, session, {"kind": type(exc).__name__, "message": str(exc)})
```

### Layer 2: Tool handler (dict returns)

Tool handlers never raise (except `Cancelled`). Errors are returned as dicts
with `ok: False` or `is_error: True`. Engine sees a successful tool call;
the LLM reads the error in the output and decides what to do.

```python
return {"ok": False, "error": f"skill '{name}' not found"}
```

### Layer 3: Event subscriber (swallow + log)

Subscriber errors are caught in `Engine._emit` and logged. A broken tracer
must never crash the engine.

```python
except Exception as exc:
    _logger.warning("subscriber %r failed: %r", subscriber.name, exc)
```

---

## Forbidden Patterns

- `except Exception: pass` with no logging (was found by review, now all replaced with `_logger.debug`)
- Raising from inside a tool handler (breaks the tool_calls/tool_result protocol)
- Catching `Cancelled` inside a tool handler without re-raising (prevents cancel propagation)
- Using `str(exc)` in user-visible output without sanitization (may leak internal paths/secrets)

---

## Common Mistakes

1. **auto_compact destroyed messages on summary failure.** Root cause: `_summarize`
   returned a placeholder string instead of raising. Fix: raise on failure, caller
   returns `did_compact=False` to preserve originals.
2. **Bare `except: pass` in Langfuse tracer** created a silent observability blackhole.
   Fix: all replaced with `_logger.debug(exc_info=True)`.
3. **MCP tool errors looked like success** because the bridge returned a dict (layer 2)
   but `ToolResult.is_error` stayed False (layer 1). This is by design but must be
   documented: MCP semantic errors live in the output dict, not in engine-level is_error.
