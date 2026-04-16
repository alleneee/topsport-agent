# Project Learnings

## Langfuse Python SDK v3 has a non-context-manager API

**Context:** Wiring LangfuseTracer as an EventSubscriber in `observability/langfuse_tracer.py`.

**Learned:** v3 exposes `langfuse.start_observation(name=..., as_type=..., input=...)`
that returns a span object you can later close with `.end()`. This is distinct from
`start_as_current_observation` which is a `with`-block helper. The explicit form is
mandatory for event-driven runtimes because span lifecycles cross multiple async events
(e.g. `LLM_CALL_START` and `LLM_CALL_END` are separate yields — you cannot wrap them
in a single `with` block).

v4 will unify `start_span`/`start_generation` into `start_observation`. Prefer
`start_observation(as_type=...)` now to stay forward-compatible.

Supported `as_type`: `span`, `generation`, `tool`, `agent`, `chain`, `retriever`,
`embedding`, `evaluator`, `guardrail`.

**Evidence:** `src/topsport_agent/observability/langfuse_tracer.py`,
`tests/test_langfuse_tracer.py`.

---

## Async generator `finally` blocks cannot `yield`

**Context:** Adding RUN_END event at the end of `Engine.run()`.

**Learned:** Python async generator `finally` blocks run during `GeneratorExit`
propagation. Attempting to `yield` inside `finally` raises. To emit a final event
regardless of success/cancel/error, wrap the inner logic in a separate async generator
and let the outer generator `yield` the RUN_END after the inner one is exhausted.

Structure:

```python
async def run(self, session):
    yield run_start_event
    async for event in self._run_inner(session):
        yield event
    yield run_end_event  # after try/except paths inside _run_inner resolve
```

`_run_inner` handles `Cancelled` / `Exception` internally and emits corresponding
events before returning. RUN_END is always reached because the inner generator
always terminates (never re-raises past the wrapper).

**Evidence:** `src/topsport_agent/engine/loop.py:165-206`.

---

## Pyright's optional-dependency escape hatch

**Context:** `langfuse` is an optional dependency group. Hard `from langfuse import X`
causes Pyright to report `reportMissingImports` when the package is not installed,
even though the code is behind a `try/except ImportError`.

**Learned:** Pyright statically analyzes `from X import Y` and
`importlib.import_module("literal")`, but it does **not** follow
`importlib.import_module(variable)`. Indirect the module name through a local
variable:

```python
module_name = "langfuse"
langfuse_module = importlib.import_module(module_name)
Langfuse = langfuse_module.Langfuse
```

Cost: `self._client` becomes `Any`. Acceptable for duck-typed tracers.

Alternative (not chosen because of the "no comments" rule): `# pyright: ignore[reportMissingImports]`.

**Evidence:** `src/topsport_agent/observability/langfuse_tracer.py:30-44`.

---

## Fire-and-forget event dispatch breaks trace ordering

**Context:** User specified fire-and-forget subscriber dispatch for minimum latency.

**Learned:** Creating a task per event (`asyncio.create_task(sub.on_event(e))`)
does not guarantee delivery order. For a Langfuse tracer this causes span lifecycle
inversion: `end()` arriving before `start_observation()` produces time-inverted
parent/child relationships and broken trace trees.

Resolution: sequential `await` inside `Engine._emit`, with per-subscriber exception
swallowing (`try/except Exception → logger.warning`). This is still effectively
fire-and-forget because Langfuse v3 SDK is non-blocking (OTEL batching internally) —
each `start_observation/update/end` call returns in microseconds.

If a subscriber genuinely blocks, the correct fix is a per-subscriber serialized
`asyncio.Queue` + consumer task, not true fan-out.

**Evidence:** `src/topsport_agent/engine/loop.py:63-74`,
`tests/test_event_subscribers.py::test_subscriber_receives_every_event_in_order`.

---

## Ephemeral context must not persist into session.messages

**Context:** Memory / skill / MCP prompt injectors attach as `ContextProvider`.

**Learned:** If injector output is appended to `session.messages`, the next step
re-injects plus sees the previous injection, doubling the block each turn. Keep
injected messages ephemeral: merge with `session.messages` at LLM-call time, do
not mutate the session.

Also collapse all `SYSTEM`-role ephemeral messages + `session.system_prompt` into
a single system block per call — Anthropic only allows one top-level system, and
OpenAI tolerates one.

**Evidence:** `src/topsport_agent/engine/loop.py:94-114` (`_build_call_messages`),
`tests/test_engine_hooks.py::test_context_provider_injected_but_not_persisted`.

---

## Tool-source deduplication: builtin wins

**Context:** MCP tool bridges and other dynamic tool sources can collide with
builtin tool names.

**Learned:** On name collision, drop the dynamic tool. Rationale: builtin tools
are audited and controlled; a misconfigured MCP server that exports a `save_memory`
or `shell` tool could otherwise shadow the safe local implementation.

**Evidence:** `src/topsport_agent/engine/loop.py:76-86` (`_snapshot_tools`),
`tests/test_engine_hooks.py::test_tool_source_dedup_prefers_builtin`.

---

## Anthropic Agent Skills frontmatter is single-line YAML

**Context:** Building `SkillRegistry` parser. Every Claude official skill in
`~/.claude/skills/` was surveyed for frontmatter variants.

**Learned:** In practice all official Claude skills use simple single-line
`key: value` frontmatter. None use YAML folded (`>`) or literal (`|`) multi-line
scalars. The minimal required keys are `name` (kebab-case) and `description`.
Other keys observed in the wild: `version`, `argument-hint`, `license`. A
~15-line hand-rolled parser (no `pyyaml` dependency) handles 100% of real skills.

Key parser pitfall: keys with hyphens (`argument-hint`) must be preserved
verbatim — do not normalize to underscores.

**Evidence:** `src/topsport_agent/skills/_frontmatter.py`,
`tests/test_skills.py::test_registry_parses_real_claude_official_skills`
(integration test against `~/.claude/skills/`).

---

## Session-scoped activation uses list, not set

**Context:** `SkillMatcher` tracks which skills are currently activated per
session.

**Learned:** Use `dict[str, list[str]]` not `dict[str, set[str]]` even though
duplicates must be prevented. Two reasons:
1. Insertion order is meaningful — injected skill bodies appear in the order
   they were activated, which keeps LLM context stable and KV-cache friendly.
2. `set` iteration order in CPython is stable within a process but not across
   processes, making regression debugging harder.

Duplicate prevention is explicit: `if name not in active: active.append(name)`.

**Evidence:** `src/topsport_agent/skills/matcher.py:12-20`.

---

## MCP Python SDK renamed `streamablehttp_client` to `streamable_http_client`

**Context:** Writing `_make_real_session_factory` for HTTP transport in
`mcp/client.py`.

**Learned:** Between MCP Python SDK v1 and v2, the HTTP client helper was
renamed from `streamablehttp_client` (one word) to `streamable_http_client`
(underscore-separated). v2 also dropped the third tuple element `get_session_id`
from the return value and moved `headers`/`timeout`/`auth` configuration to an
external `httpx.AsyncClient` passed via the `http_client=` parameter.

Migration snippet:

```python
# v1 (deprecated)
async with streamablehttp_client(url, headers={...}, timeout=30) as (r, w, get_id):
    ...

# v2
http_client = httpx.AsyncClient(headers={...}, timeout=30)
async with http_client:
    async with streamable_http_client(url=url, http_client=http_client) as (r, w):
        ...
```

Use Context7 to verify every library version before writing client code —
SDK renames like this are invisible in training data.

**Evidence:** `src/topsport_agent/mcp/client.py:79-108`.

---

## MCP sessions cannot safely cross asyncio tasks

**Context:** Initial plan was to keep a long-lived `ClientSession` open across
multiple engine runs for efficiency. Testing revealed broken cancel scopes.

**Learned:** MCP's `stdio_client` and `ClientSession` are built on nested
`async with` blocks that rely on `anyio` cancel scopes. These scopes are bound
to the asyncio task that entered them. If you `enter_async_context` in task A
and `__aexit__` in task B (e.g. via a shared `AsyncExitStack`), anyio raises
`RuntimeError: Attempted to exit cancel scope in a different task`.

**Workaround**: open a fresh session per call (`call_tool`, `get_prompt`,
`read_resource`) via `async with self._session_factory() as session:`. Cache
only the immutable **list** results (`list_tools`, `list_prompts`,
`list_resources`) in memory.

Cost: each tool invocation incurs the transport startup (for stdio, a fork).
Benefit: every call is task-local, cancel-safe, and exception-safe without
reference-counting sessions.

If startup cost becomes a bottleneck, the correct fix is a dedicated
consumer task per client with an `asyncio.Queue` for request dispatch —
**never** share a raw session across tasks.

**Evidence:** `src/topsport_agent/mcp/client.py:24-60`,
`tests/test_mcp.py::test_mcp_client_lazy_caches_tool_list`.

---

## Test-injectable factories let optional deps stay optional

**Context:** `mcp` and `langfuse` are optional dependency groups. Tests for
`MCPClient`, `MCPToolSource`, and `LangfuseTracer` must run without installing
either package.

**Learned:** Pattern: core classes take a **factory callable** (or pre-built
client object) as a constructor parameter. A separate `@classmethod from_config`
alternate constructor lazily imports the optional dependency.

```python
class MCPClient:
    def __init__(self, name: str, session_factory: SessionFactory) -> None: ...

    @classmethod
    def from_config(cls, config: MCPServerConfig) -> "MCPClient":
        return cls(config.name, _make_real_session_factory(config))
```

Tests build the class directly with a mock factory; production calls
`from_config`. The module can be imported without the optional dep — only
`from_config` triggers `importlib.import_module`.

Combined with variable-indirected `importlib.import_module(var_name)` (see
earlier learning), this means Pyright stays quiet and tests stay isolated.

**Evidence:** `src/topsport_agent/mcp/client.py:14-25`,
`src/topsport_agent/observability/langfuse_tracer.py:16-56`,
`tests/test_mcp.py` (15 tests with zero `mcp` package imports).

---

## Two-level error model: protocol success vs. semantic failure

**Context:** MCP has its own `isError` flag on `CallToolResult`, distinct from
exceptions raised during the protocol call itself.

**Learned:** Three failure modes must be kept separate:

| Layer | Example | Runtime handling |
| --- | --- | --- |
| Transport / protocol | stdio subprocess died, HTTP 500, JSON-RPC error | `handler` catches exception, returns `{"is_error": True, "error": "..."}` |
| MCP semantic (server-reported) | Tool ran, returned `isError=True` with error text | `handler` returns `{"is_error": True, "text": "..."}` |
| Engine tool-call layer | Handler itself crashed or unknown tool | `ToolResult.is_error = True` set by engine |

The engine's `ToolResult.is_error` is reserved for the third layer. MCP's two
layers both surface into the `output` dict, letting the LLM see the distinction
but keeping the engine invariant clean: if `ToolResult.is_error` is True, the
engine itself knows something went wrong; otherwise the LLM is responsible for
interpreting the payload.

**Evidence:** `src/topsport_agent/mcp/tool_bridge.py:44-74`,
`tests/test_mcp.py::test_mcp_tool_handler_reports_tool_exception`.

---

## Claude Desktop config format is a free standard

**Context:** Designing the JSON schema for MCP server configuration.

**Learned:** Claude Desktop's `claude_desktop_config.json` uses `mcpServers`
as the top-level key with server names as sub-keys. Keeping this shape means
any existing Claude Desktop configuration file is directly usable without
translation — no separate config format to maintain, no migration tooling.

```json
{
  "mcpServers": {
    "<name>": {
      "transport": "stdio" | "http",
      "command": "...",          // stdio only
      "args": [...],             // stdio only
      "env": {...},              // stdio only
      "url": "...",              // http only
      "headers": {...},          // http only
      "timeout": 30              // http only
    }
  }
}
```

Principle: when an ecosystem has a de-facto config format, adopt it instead of
inventing a new one. The cost is zero; the interop benefit is large.

**Evidence:** `src/topsport_agent/mcp/config.py`,
`tests/test_mcp.py::test_load_mcp_config_stdio_and_http`.
