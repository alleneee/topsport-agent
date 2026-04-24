# Project Learnings

## Lazy-import factory: `try` must wrap the constructor, not the submodule import

**Context:** `create_database` dispatches backends via
`importlib.import_module("topsport_agent.database.backends.postgres")` and then
`mod.PostgresGateway(config)`. The expected UX is that a missing `asyncpg`
turns into a helpful `ImportError("... install via: uv sync --group db")`.
Initial factory only wrapped the `import_module(submodule_name)` call in the
`try`; a final reviewer flagged that this wrapper is always a no-op.

**Learned:** The project's optional-dep pattern has `postgres.py` **import
asyncpg lazily inside `PostgresGateway.__init__`** via
`importlib.import_module(mod_name)`. That means:

- `importlib.import_module("topsport_agent.database.backends.postgres")`
  always succeeds — postgres.py has no top-level `asyncpg` reference.
- The real `ImportError` fires on `mod.PostgresGateway(config)` when
  `__init__` runs `importlib.import_module("asyncpg")`.

Therefore `try` must wrap **both** the `import_module(submodule)` call and
the subsequent `ClassName(config)` construction. Wrapping only the first
lets the raw `ModuleNotFoundError: No module named 'asyncpg'` escape past the
translate-to-friendly-error layer.

Regression test pattern: `monkeypatch.setitem(sys.modules, "asyncpg", None)`
then expect the friendly-message `ImportError`.

**Evidence:** `src/topsport_agent/database/factory.py` (commit `a33dddf`
fixed the try-block scope), `tests/test_database_factory.py::test_factory_postgres_missing_asyncpg_gives_friendly_import_error`.

---

## `uv sync --group X` is destructive: it removes every group not listed

**Context:** A subagent installing the `db` group for Postgres tests ran
`uv sync --group db`. Later subagents discovered `tests/test_permission_*.py`
failed to collect: `ModuleNotFoundError: No module named 'fastapi'`. The
permission tests were fine — fastapi had been uninstalled.

**Learned:** `uv sync --group X` installs X **and** removes all other groups
from the venv, unless you re-list them. The project has 9 optional groups
(`dev db api mcp metrics llm sandbox tracing browser`); any test depending
on `fastapi` (the `api` group) breaks after a partial sync.

Safe recipes:
- `uv sync --all-groups` — simplest, reinstalls everything
- `uv sync --group dev --group db --group api --group mcp --group metrics --group llm --group sandbox --group tracing --group browser` — explicit list
- Don't sync at all if the needed group is already installed

Prompt-level fix: when dispatching subagents that might sync, tell them the
venv already has all groups installed and that a partial sync will break
unrelated tests.

**Evidence:** Commit `0a93f20` followed by `uv run pytest` showing 4
collection errors on `test_permission_*` / `test_sandbox` due to missing
`fastapi`. Restored via full-group resync.

---

## Subagents constrained by `.trellis/workflow.md` can't `git commit`

**Context:** While running subagent-driven development to execute the
database skeleton plan, each TDD task's Plan-defined Step 5 is "commit".
The `implement` subagent type read `.trellis/workflow.md` and refused:
"AI should not commit code". It reported `DONE_WITH_CONCERNS` but the
working tree was still modified/untracked.

**Learned:** Project-level rules (`.trellis/workflow.md`, global `CLAUDE.md`)
override per-task instructions that the main agent writes into subagent
prompts. For any plan whose tasks include commits, the main agent must:

1. Explicitly tell the subagent in its prompt: "Do NOT `git commit`. Stop
   before commit. The main agent will commit after verifying."
2. Run `git add` + `git commit` itself after verifying the subagent's work.

Without the explicit "no commit" instruction, subagents that encounter the
project rule will produce a confusing "DONE - Ready for commit" report
(code ready but not committed) that doesn't match any of the four documented
statuses. Don't trust a DONE without verifying with `git status`.

**Evidence:** First Task 0 dispatch to `implement` subagent produced a
"DONE" report while `git status` showed 5 untracked/modified files. Fixed
by amending all subsequent Plan A dispatch prompts to say "Do NOT commit".

---

## "Enterprise ACL" can look complete per-module and still be functionally disconnected

**Context:** After landing the capability-ACL subsystem
(`PermissionFilter / Persona / Assignment / AuditLogger / KillSwitch`),
every unit test passed and each module looked correct in isolation. An
external code reviewer (codex) still called the system "假的" (not real)
and backed the claim with a punch-list of integration gaps. Every gap
was real on inspection; none were caught by the existing test suite.

**Learned:** Retrofit ACLs onto an existing framework fail at **boundaries**,
not at the enforcement logic. Audit these four planes every time:

1. **Control plane → execution plane (binding):** Admin CRUD for Personas
   existed; Assignments (tenant/user/group → persona) had no HTTP path.
   Operators had no way to actually *grant* capabilities. Lesson: any
   runtime enforcement target (`session.granted_permissions`) must have a
   corresponding HTTP/admin write path that hits the same field.

2. **Default session factory:** `SessionStore.get_or_create` called
   `agent.new_session()` (sync). Persona resolution only happened in
   `new_session_async()`. So in the production server path, the ACL was
   never invoked even when fully configured. Lesson: multiple session
   factories = multiple places to get the ACL wrong. Fold them all
   through one path, or pick the async one as canonical.

3. **Delegation plane:** `Agent.spawn_child` forwarded `context_providers /
   tool_sources / post_step_hooks / event_subscribers / sanitizer` but
   dropped `permission_filter / audit_logger / permission_checker /
   permission_asker` and the session's `granted_permissions / tenant_id /
   principal / persona_id`. The parent agent was ACL-gated; the child
   bypassed everything. Lesson: delegation inherits code-path intent,
   which means **all** capabilities, not just the ones the original
   author remembered.

4. **Default config bypass:** `default_agent()` hardcoded
   `enable_skills=True, enable_memory=True, enable_plugins=True` even
   though `ServerConfig` had gates for them. The server's config never
   reached the agent factory. Lesson: a config field that exists but
   isn't threaded into the factory is worse than a missing field —
   operators *think* they've closed a door that's still open.

There's also a time-bomb subpattern: **public identifier shapes colliding
with internal validators**. `namespace_session_id(principal, hint)`
produced `principal::hint`; `FileMemoryStore._SAFE_ID_RE = ^[a-zA-Z0-9._-]+$`
rejected the colon. Default chain enabled memory injection every step.
Bomb triggers on first session with a non-empty hint. Lesson: every
public-facing identifier producer must be cross-referenced against every
internal consumer's format check, and the cross-check belongs in a test.

**How to spot these gaps without an external reviewer:**
- Grep for every place where `granted_permissions / permission_filter /
  audit_logger` is read — each must have a populator or be assert-covered
  as explicitly defaulted.
- For every `ServerConfig` flag, grep forward to the factory that should
  respect it; if the flag doesn't appear there, it's dead config.
- For every identifier transformer (`namespace_*`, `quote_*`), grep
  forward to every format validator it feeds; mismatches are bombs.

**Evidence:** `tests/test_permission_holistic_wiring.py` (15 tests
covering all 6 gaps), fix sweep landed 2026-04-23 covering
`server/app.py`, `server/sessions.py`, `server/permission_api.py`,
`agent/base.py`, `agent/default.py`, `memory/file_store.py`,
`tools/file_ops.py`, `memory/tools.py`, `plugins/agent_registry.py`.

---

## Capability-based ACL beats runtime decision engines for enterprise agent platforms

**Context:** Designing the permission subsystem for an enterprise internal
AI productivity platform where the sandbox already confines risky operations.
Initial design was a runtime ALLOW/DENY/ASK decision engine with pattern
rules and pending-approval workflow.

**Learned:** When risky operations are already confined (sandbox), runtime
decisions are asking the wrong question. The real question is "which tools
should this agent see at all?" — a static capability check at snapshot time,
not a per-call pattern match.

Signs the problem is capability-ACL rather than runtime-decision:
1. Operations are trigger-and-leave (no user watching to approve)
2. Risky work is already sandboxed (or containerized)
3. Tool populations are stable (not generated per session)
4. Tenants are well-defined (not per-operation trust decisions)

Capability-ACL characteristics:
- Tool declares required_permissions (static, at definition time)
- Subject carries granted_permissions (populated at session creation)
- Runtime = `tool.required.issubset(subject.granted)` — O(1) per tool
- No ASK, no pending, no mutation — decisions happen at configuration time

Sunk-cost temptation: the previous runtime-decision module had tests and
working code, so "why not keep it and add capability-ACL alongside?" The
right call was deprecate + replace in one minor cycle, because runtime-
decision surface is a footgun: future contributors would wire Asker/Checker
into code paths that should use the Filter, and the architecture would
bifurcate.

**Evidence:** `docs/superpowers/specs/2026-04-23-permission-capability-acl-design.md`,
`docs/superpowers/plans/2026-04-23-permission-capability-acl.md`,
`src/topsport_agent/engine/permission/filter.py`.

---

## TypeVar bound to `BaseModel` preserves type inference in factory classmethods

**Context:** `ToolSpec.from_model(input_model=MyInput, handler=my_handler)` was
initially typed as `handler: Callable[[BaseModel, ToolContext], Awaitable[Any]]`.
Pyright rejected every call site because concrete subclasses (`_SearchInput`)
are not parameter-level substitutable for `BaseModel` — callable parameters are
contravariant.

**Learned:** Use a bounded TypeVar so the factory binds `T = MyInput` for that
specific call:

```python
from typing import TypeVar
_BM = TypeVar("_BM", bound="BaseModel")

@classmethod
def from_model(
    cls, *,
    input_model: type[_BM],
    handler: Callable[[_BM, ToolContext], Awaitable[Any]],
    **kwargs,
) -> "ToolSpec": ...
```

Now `input_model=MySearchInput` forces `_BM=MySearchInput`, and the handler
parameter type flows through — IDE completion and mypy/Pyright work. Without
the TypeVar, every caller has to annotate their handler as `BaseModel` and
lose field-level types in the body.

**Evidence:** `src/topsport_agent/types/tool.py::ToolSpec.from_model`,
`tests/test_toolspec_pydantic.py::test_handler_receives_typed_pydantic_instance`.

---

## Fail-closed is the only safe default for permission machinery

**Context:** Designing the permission subsystem (`types/permission.py` +
`engine/permission.py`). Four failure modes exist: checker not configured,
checker raises, asker not configured, asker returns ASK (contract violation).

**Learned:** Every single non-ALLOW path must default to DENY. Dropping any
one of these gates opens a "permissive by accident" hole:

| Failure mode | Wrong default | Right default |
|---|---|---|
| `permission_checker=None` | block everything | **let it through** (opt-in API: no checker means no policy, back-compat) |
| checker raises exception | retry / allow | **DENY** with "checker error: ..." |
| checker returns ASK, no asker | treat as ALLOW | **DENY** "no asker configured" |
| asker raises exception | allow | **DENY** with "asker error: ..." |
| asker returns ASK | loop | **DENY** "contract violation" |

The first row is the only "allow by default" — and only because the checker
itself is explicitly opt-in. Once the user has configured a checker, ALL
other failure modes deny.

Also: `PermissionDecision` must be `frozen=True`. Non-frozen decisions let a
rogue subscriber mutate `decision.behavior` from ALLOW to DENY (or vice versa)
between the checker and the handler call.

**Evidence:** `src/topsport_agent/engine/loop.py::_invoke_tool` (permission block),
`tests/test_permission.py::test_checker_exception_treated_as_deny`,
`tests/test_permission.py::test_destructive_denied_when_no_asker`.

---

## `dataclasses.replace` is the correct tool for ToolSpec decorators

**Context:** `tools/executor.py::ToolExecutor.wrap` was hand-constructing a new
`ToolSpec(name=..., description=..., parameters=..., handler=...)` to wrap the
handler. This silently dropped every field not listed in that constructor call —
the original bug was dropping `trust_level`. When we added `read_only`,
`destructive`, `concurrency_safe`, `max_result_chars`, `validate_input`, a hand-
constructed wrapper would have dropped all five.

**Learned:** Always use `dataclasses.replace(spec, handler=new_handler)` for
decorator-style wrappers over dataclasses. It copies every field by default,
so future field additions to `ToolSpec` flow through wrappers automatically.
Hand-constructing is fragile: every new field is a landmine for every
pre-existing wrapper.

Write a regression test that asserts all metadata fields survive wrapping —
we caught the trust_level loss only after noticing sanitizer was suddenly
direct-passing untrusted results, which took longer than it should have.

**Evidence:** `src/topsport_agent/tools/executor.py:wrap`,
`tests/test_toolspec_extended.py::test_tool_executor_wrap_preserves_all_fields`.

---

## Pre-scheduling with asyncio.create_task inside an async generator

**Context:** Engine's `_execute_tool_calls` is an async generator that yields
`TOOL_CALL_START/END` events in `calls`-list order — tests and tracers depend
on this order. But we also wanted concurrent execution for `concurrency_safe`
tools. Running `asyncio.gather` would break the event ordering guarantee.

**Learned:** The clean pattern is **pre-schedule as tasks, await in order**:

```python
# Phase 1: pre-schedule all concurrency_safe handlers as background tasks
scheduled: dict[int, asyncio.Task] = {}
for idx, call in enumerate(calls):
    tool = _find_tool(call.name, pool)
    if tool and tool.concurrency_safe:
        scheduled[idx] = asyncio.create_task(_invoke_tool(call, tool, session))

# Phase 2: iterate in original order, yielding events; await if pre-scheduled
for idx, call in enumerate(calls):
    yield TOOL_CALL_START_event(...)
    result = await scheduled[idx] if idx in scheduled else await _invoke_tool(...)
    yield TOOL_CALL_END_event(...)
```

Handlers run in parallel (wall-clock win), but events still emit in the original
call order (observability invariant preserved). Cancellation in the middle of
the loop must `task.cancel()` the remaining scheduled tasks or they leak into
the event loop.

Never let yourself reorder events by completion time "for performance" —
subscribers (tracer, metrics, server SSE) rely on the causal order, and
debugging out-of-order event streams is far worse than losing a few ms.

**Evidence:** `src/topsport_agent/engine/loop.py::_execute_tool_calls`,
`tests/test_toolspec_extended.py::test_concurrency_safe_tools_run_in_parallel`,
`tests/test_toolspec_extended.py::test_mixed_safe_and_unsafe_in_same_batch`.

---

## Parallel schema layer: pydantic `extra="ignore"` for backwards-compatible typing

**Context:** Borrowing claude-code's Zod `discriminatedUnion` pattern for `Event.payload`
in `types/event_payloads.py`. 20+ call sites already read `event.payload.get(...)` as
untyped dict; a one-shot migration to strict types was too risky.

**Learned:** Pydantic's `ConfigDict(extra="ignore", frozen=True)` is the right escape
hatch for adding a typed access layer to an existing dict payload without breaking
existing consumers:

- `extra="ignore"` silently drops fields the schema doesn't declare → publishers can
  keep adding debug/telemetry fields without coordinating with every subscriber
- `frozen=True` prevents cross-subscriber mutation after validation
- The original `payload: dict[str, Any]` stays untouched — old code paths continue
  to work, new code paths opt in via `event.typed_payload()`

Key design choice: **don't** use `extra="forbid"` here. Forbid surfaces publisher
drift as validation errors in every subscriber, which breaks the gradual-adoption
story. `ignore` makes the schema a **minimum contract** for subscribers rather than
a maximum contract for publishers.

Strictness is preserved where it matters: required fields missing → `ValidationError`,
type coercion failures (e.g. `"not-a-bool"` for `is_error`) → `ValidationError`.
Assignment on returned model → `ValidationError` due to `frozen=True`.

**Evidence:** `src/topsport_agent/types/event_payloads.py`, `src/topsport_agent/types/events.py:55-73`,
`tests/test_event_payloads.py`.

---

## Don't inherit `Protocol` in test doubles — duck-type instead

**Context:** Writing a `_ScriptedProvider` test fixture that satisfies `LLMProvider`.

**Learned:** Python's `typing.Protocol` is structural by design — implementers should
**not** `class Foo(LLMProvider)` inherit it. Pyright treats fields declared on the
Protocol (like `name: str`) as abstract when you inherit, and reports
`reportAbstractUsage` unless every Protocol attribute is explicitly implemented.

Duck-type instead:

```python
class _ScriptedProvider:            # no inheritance
    name = "scripted"               # plain class attr satisfies the Protocol
    async def complete(self, request): ...
```

Engine code only sees `LLMProvider` through structural checks
(`isinstance(provider, StreamingLLMProvider)` with `@runtime_checkable`), so
duck-typed classes pass at runtime and at type-check time.

**Evidence:** `tests/test_event_payloads.py::_ScriptedProvider`,
`src/topsport_agent/llm/provider.py:11-16`.

---

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

---

## Playwright click fails in nested iframes with "outside of the viewport"

**Context:** Automating a JD merchant backend demo where target buttons live in
a deeply nested iframe (`shop.jd.com/jdm/legacy/...` hosts `mc.jd.com/...`
inside it). `Locator.click()` on the inner-frame button kept retrying with
`element is visible, enabled and stable — scrolling into view if needed — done
scrolling — element is outside of the viewport` until timeout, even after
`scroll_into_view_if_needed`.

**Learned:** Playwright's click uses the outer page's viewport to judge
visibility. For elements inside a cross-origin child frame, the coordinate
transform between frame and page can leave the element perpetually "outside"
from the engine's point of view. Workaround: trigger the click inside the
frame via raw DOM:

```python
await target_frame.evaluate("""() => {
  const btns = [...document.querySelectorAll('button')].filter(
    b => (b.innerText||'').trim() === '立即报名'
  );
  btns[0].scrollIntoView({behavior:'instant', block:'center'});
  btns[0].click();
}""")
```

This trades Playwright's actionability guarantees for a JS call — fine for
exploratory automation, risky for production test suites (no auto-wait).

Also important: when selecting the right child frame, **use
`fr.url.startswith("https://mc.jd.com")` not `"mc.jd.com" in fr.url`** — the
wrapper page's URL contains the child origin in its path, so `in` matches
both frames and picks the wrong one.

**Evidence:** `scripts/jd_demo/stage5_signup.py`,
`.claude/skills/jd-merchant-activity-signup/SKILL.md`.

---

## `uv run python` silently buffers stdout when backgrounded

**Context:** Launching a Playwright demo script via Bash `run_in_background=true`;
Python `print` output never appeared in the log file, making it look like the
script hung.

**Learned:** When stdout is not a TTY (any pipe/file/background redirect),
CPython switches to block buffering. The process was fine — its prints were
sitting in the buffer. Fix: export `PYTHONUNBUFFERED=1` (or run `python -u`,
or `sys.stdout.reconfigure(line_buffering=True)`). `uv run` doesn't add
`-u` automatically.

Diagnostic tell: if the file is 0 bytes but `ps` shows the process is still
running and doing work, it's buffering, not hanging. Don't kill it.

**Evidence:** `scripts/jd_demo/` (all stages use `PYTHONUNBUFFERED=1` in the
bash invocation).
