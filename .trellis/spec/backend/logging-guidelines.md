# Logging Guidelines

> How observability works in this project.

---

## Overview

Standard library `logging` module. Each module creates its own logger via
`_logger = logging.getLogger(__name__)`. No third-party logging framework.

Primary observability is event-based: `Engine.run()` yields `Event` objects
that `EventSubscriber` implementations (like `LangfuseTracer`) consume.
Python logging is secondary, used for internal diagnostics.

---

## Log Levels

| Level | When |
|-------|------|
| `WARNING` | Event subscriber failed, MCP list_tools failed, non-critical operational issue |
| `INFO` | auto-compaction triggered (CompactionHook) |
| `DEBUG` | Langfuse SDK call failures (all `except Exception` blocks in tracer) |

No `ERROR` level logs from the runtime itself. Engine errors are surfaced as
`EventType.ERROR` events, not as log messages.

---

## What to Log

- Subscriber dispatch failures (`engine/loop.py:_emit`)
- MCP tool source list_tools failures (`mcp/tool_bridge.py`)
- Compaction decisions (CompactionHook)
- Langfuse SDK operation failures (all handler methods)

---

## What NOT to Log

- Session message content (may contain user data, API keys, PII)
- Tool call arguments (may contain sensitive parameters)
- API keys, tokens, secrets (even partial)
- Full exception stack traces at WARNING level (use DEBUG for traces)
- Event payloads (these go to EventSubscribers, not to logs)
