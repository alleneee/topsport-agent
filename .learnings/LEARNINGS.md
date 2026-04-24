# Project Learnings

## `list["ForwardRef"]` double-stringifies under `from __future__ import annotations`

**Context:** Adding `content_parts: list["ContentPart"] | None = None` to the
`Message` dataclass in `types/message.py`. Runtime tests passed (15/15) but
Pyright reported:

```
Cannot access attribute "content_parts" for class "Message"
No parameter named "content_parts"
```

`ContentPart = TextPart | ImagePart | VideoPart` was already defined at
module top-level, above `Message`. The forward-ref quoting looked defensive
— shouldn't it be fine either way?

**Learned:** When `from __future__ import annotations` is active (line 1 of
the module), **all** type annotations are already lazy-evaluated strings at
runtime. Wrapping a name in extra `"..."` produces a double-quoted string
that Pyright can't resolve as a type reference, even when the name exists
in module scope. The runtime is happy (dataclass just stores a string
annotation either way), but Pyright's static analyzer treats the field as
type-unknown, and any downstream code that reads `msg.content_parts`
cascades into `reportAttributeAccessIssue` noise.

Fix: drop the quotes when the name is defined above the reference point.

```python
# wrong (under __future__.annotations): Pyright can't resolve
content_parts: list["ContentPart"] | None = None

# right: PEP 563 lazy-strings the annotation for you
content_parts: list[ContentPart] | None = None
```

Rule of thumb: only use string forward-refs when the referenced name is
defined **below** the annotation (genuine forward reference). If it's above,
let `from __future__ import annotations` do the deferral.

**Evidence:** `src/topsport_agent/types/message.py::Message.content_parts`
(commit `033f719` introduced double-quote, immediately fixed in follow-up
edit before merge). Regression surface would be caught by any Pyright run
on files that consume `Message.content_parts`.

---

## Middleware shim pattern for lifespan-built dependencies

**Context:** FastAPI's `app.add_middleware(MiddlewareCls, ...)` wires
middleware at app-creation time, but the real `RedisSlidingWindowLimiter`
requires an async `SCRIPT LOAD` that only happens in `lifespan()`. Two
canonical solutions (pass limiter factory; or build middleware inside
lifespan and monkey-patch the stack) are both noisy.

**Learned:** The cleanest pattern is a lazy shim:

```python
class _LazyLimiter:
    async def check(self, rules):
        real = getattr(app.state, "ratelimit_limiter", None)
        if real is None:
            return RateLimitDecision(allowed=True, ...)  # allow-all
        return await real.check(rules)

app.add_middleware(RateLimitMiddleware, limiter=_LazyLimiter(), ...)
```

The shim is instantiated at `add_middleware` time, but every `check()` call
pulls the live limiter from `app.state.ratelimit_limiter` — which lifespan
populates before FastAPI starts accepting requests. The allow-all branch is
only reachable in non-standard embedding scenarios (e.g. ASGI mount without
lifespan). Worth a log-once warning if it triggers.

This generalises to ANY middleware that needs a dependency built inside
lifespan: DB connection, feature flag client, cache, metrics exporter.
Don't try to "build it eagerly in create_app" — lifespan exists precisely
because some things must be async-initialised.

**Evidence:** `src/topsport_agent/server/app.py::_LazyLimiter`,
`tests/test_server_lifespan_ratelimit.py::test_disabled_ratelimit_does_not_touch_redis`.

---

## `prometheus_client.Counter` is process-global; use private registries per test

**Context:** Running `pytest` with multiple `RateLimitMetrics()` fixtures
threw `ValueError: Duplicated timeseries in CollectorRegistry:
{'ratelimit_requests_total'}`. Each fixture tried to register the same
counter on the default (process-global) registry — fine in production,
fatal in tests.

**Learned:** When building metrics classes that might be instantiated
multiple times in one process (tests, worker pools, multi-tenant embeds),
accept an optional `registry` arg **and default to a fresh
`CollectorRegistry()` instead of the global one**:

```python
def __init__(self, *, registry: Any | None = None) -> None:
    if registry is None:
        registry = prom.CollectorRegistry()
    self._requests = prom.Counter(..., registry=registry, **kwargs)
```

Production callers who want metrics exposed on `/metrics` pass the real
process-global registry (`prom.REGISTRY`) explicitly. Library code should
never default to `prom.REGISTRY` implicitly — that makes the class
single-instance-per-process, which is a hidden footgun.

The related idiom in the codebase is the `metrics: X | None = None` +
"build a fresh one if missing" pattern used by
`RateLimitMiddleware.__init__`.

**Evidence:** `src/topsport_agent/ratelimit/metrics.py::RateLimitMetrics.__init__`,
`tests/test_ratelimit_metrics.py::test_metrics_with_prometheus_registers_counters`
(explicitly passes a fresh `CollectorRegistry()`).

---

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

---

## Anthropic-compatible endpoints stringify nested object tool-call values

**Context:** Wiring a shared `ServerKVContext` (subclass of `PlanContext`) into
HTTP `/v1/plan/execute` so DAG steps can share state via `plan_context_merge` /
`plan_context_read`. When the LLM (MiniMax-M2.7 via its anthropic-compatible
endpoint) tried `plan_context_merge(key="kv", value={"title": "Example Domain"})`,
the reducer crashed with `TypeError: requires an object/dict, got str`.

**Learned:** Some third-party anthropic/openai-compatible endpoints
(MiniMax confirmed; likely some Qwen / DeepSeek setups too) **JSON-stringify
nested-object tool-call argument values** before sending them in
`tool_use.input`. The LLM writes `{"key": "kv", "value": {...}}` but what
reaches the tool handler is `{"key": "kv", "value": "{\"title\":...}"}` —
`value` is a `str`, not a `dict`. Anthropic's own API does not do this.

Same family as the OpenAI-adapter's `{"_raw_arguments": ...}` fallback
already documented elsewhere in this file: tool-call argument transport
is under-specified once you leave the reference implementation, and every
handler that accepts non-scalar argument values needs a str→json.loads
detection-and-decode pass of its own.

Canonical fix at the tool-handler / reducer layer (not the adapter — adapters
decode top-level `arguments` once but do not recurse into per-field values):

```python
def _kv_dict_merge(current: dict, update: Any) -> dict:
    if isinstance(update, str):
        try:
            update = json.loads(update)
        except ValueError as exc:
            raise TypeError(f"expected object, got non-JSON string: {exc}")
    if not isinstance(update, dict):
        raise TypeError(f"expected object, got {type(update).__name__}")
    return {**current, **update}
```

Rule: for any tool-call argument typed object/array, accept `Any`, detect
`str`, `json.loads` once, then re-validate the shape.

**Evidence:** `src/topsport_agent/server/plan.py::_kv_dict_merge`
(`plan-ctx-002` DEBUG log captured the "got str" failure; `plan-ctx-003`
confirmed the fix by successfully round-tripping "Example Domain" through
the shared KV context to a file on disk — `/tmp/plan_ctx_v3.txt`).

---

## `Agent._run_plan` must set `_engine._current_session` before orchestrator runs

**Context:** Collapsing the HTTP `/v1/plan/execute` and `/v1/chat/completions`
paths into a single `Agent.run(mode="plan", plan=...)` facade. The HTTP plan
router used to own the Orchestrator directly and constructed a fresh parent
Agent per request. Moving that inside Agent meant `Agent._run_plan` now
creates the Orchestrator with `parent_agent=self`. On first test run, the
full DAG executed but sub-step sessions silently lost `granted_permissions`
(fail-closed — no ACL tools visible), even though the parent session had
them.

**Learned:** `Agent.spawn_child` inherits tenant/principal/permissions from
either a caller-passed `parent_session` **or** `self._engine._current_session`
(the "live running session" slot that `Engine._run_inner` sets when you call
`engine.run(session)`). In the old `_run_plan`-free code path, the HTTP
request called `engine.run()` first and the slot was populated naturally. In
the facade version, `_run_plan` never calls `engine.run()` itself — it only
drives `Orchestrator.execute()`, which goes through `spawn_child`. So the
slot stays `None` and every sub-step gets `granted_permissions=∅`.

Fix: `_run_plan` must explicitly set (and clear) the slot:

```python
self._engine._current_session = session
try:
    async for event in orchestrator.execute():
        yield event
finally:
    self._engine._current_session = None
```

Generalisation: any Agent method that dispatches to a sub-engine/orchestrator
without going through the main `engine.run()` path needs to either
(a) plumb `parent_session=` explicitly through `spawn_child`, or
(b) temporarily set `_engine._current_session`. (a) is cleaner but requires
changing the Orchestrator API; (b) is a minimal patch. The presence of the
`_current_session` slot at all is a smell — it's a hidden parameter of
`spawn_child` that was fine when there was one caller (`Engine._run_inner`)
but grows brittle as more facade methods appear.

**Evidence:** `src/topsport_agent/agent/base.py::Agent._run_plan` (the
try/finally around orchestrator.execute). Regression surface: any future
facade method that composes Agent capabilities without first running the
main ReAct loop — reflection strategy, tree-of-thoughts, etc. Watch for
"granted_permissions=∅ in sub-step" as the symptom.

---

## Security tests that white-box private helpers need signature updates during facade refactors

**Context:** Same facade refactor collapsed `server/plan.py::_stream_plan`
into `server/chat.py::_stream_plan` with a different signature
(`(orchestrator, request, parent_agent)` → `(entry, plan, request)`). A
SEC-005 test (`test_plan_error_payload_does_not_leak_exception_message`)
imported the old helper directly and constructed a `FakeOrch` to drive the
error branch. It broke immediately on signature change.

**Learned:** White-box unit tests that exercise private `_helper`
functions are the right choice for security invariants — black-box testing
the error sanitization path would require triggering a real backend failure
and is flaky. But they create **legitimate coupling** between test and
private API, which means any refactor that changes the private function's
shape is a two-file change: the implementation *and* the SEC test. Don't
try to remove the coupling by making the test go through the public HTTP
path; the blast radius of a leaked `str(exc)` containing `sk-…` is high
enough to justify the coupling.

When moving private helpers across modules during a facade refactor:
1. Grep for `from X import _helper` across tests FIRST
2. Either re-export from the old location (`from .chat import _stream_plan`
   back in `plan.py`) OR update every caller in one commit
3. If the signature changes semantically, update tests too — don't contort
   the new signature to match the old test

There's a secondary point that bit me: the new code path went through
`agent.run(mode="plan")` which doesn't normally raise; the raw exception
from the underlying exec path surfaced as `str(exc)` in the SSE error
event. This is the "moved where the exception fires but forgot to update
the redaction" problem. Always re-verify the SEC invariant **end-to-end**
after moving any error-handling code, not just rely on test updates.

**Evidence:** `src/topsport_agent/server/chat.py::_stream_plan` (the
`except Exception` branch now yields `{"message": "plan execution failed",
"type": type(exc).__name__}` — deliberately drops `str(exc)` to prevent
API key leakage); test migration at
`tests/test_server_sandbox.py::test_plan_error_payload_does_not_leak_exception_message`.

---

## Agent stores subscribers/tool_sources in TWO places — runtime adds must write both

**Context:** After shipping the Langfuse tracing gate on HTTP server (opt-in
via `ENABLE_LANGFUSE=true`), a 2-step plan submitted through
`/v1/chat/completions` mode="plan" produced a chat trace in Langfuse but
**no traces for the plan's sub-steps**. The chat endpoint worked; plan mode
looked like tracing was disabled even though the global flag was on.

**Learned:** An `Agent` keeps its event subscribers / tool sources / context
providers in two synchronized-at-construction-time places:

1. `self._engine._subscribers` (and `_tool_sources`, `_context_providers`) —
   consumed by the **currently running** engine.
2. `self._capability_bundle["event_subscribers"]` (and `"tool_sources"`,
   `"context_providers"`) — consumed by `spawn_child` when it constructs
   each sub-agent's Engine.

`Engine.add_event_subscriber(sub)` / `Engine.add_tool_source(src)` only
mutate #1. #2 is a snapshot taken in `Agent.from_config`; it doesn't
observe runtime changes to the engine's lists. So any wrapper / decorator
that lazily attaches a subscriber via `agent.engine.add_event_subscriber(...)`
gets it on the **parent** agent only — the moment Plan mode (or any
spawn_child-based feature) fans out, the fresh sub-Engines are built with
the **pre-wrapper** bundle and miss the subscriber entirely.

In our case `_wrap_with_extras` / `_wrap_with_metrics` in `server/app.py`
did exactly this. The plan path silently lost observability. Tests didn't
catch it because tests build Agent directly (not through the server-side
wrapper chain), and even for the wrappers they don't assert sub-agent
telemetry.

Fix: runtime registration must write BOTH paths.

```python
agent.engine.add_event_subscriber(sub)
agent._capability_bundle.setdefault("event_subscribers", []).append(sub)
# and for tool_sources:
agent.engine.add_tool_source(src)
agent._capability_bundle.setdefault("tool_sources", []).append(src)
```

Longer-term fix: promote this to a public `Agent.register_subscriber(sub)`
method that encapsulates the two-write invariant. `_capability_bundle`
being underscore-prefixed + needing two-write synchrony is a leaky
abstraction.

Diagnostic rule of thumb: **"parent agent has the subscriber, sub-agents
don't"** is the fingerprint. If Langfuse / Prometheus / MCP bridge works
for `/v1/chat/completions` default mode but disappears for plan / delegate
/ spawn_agent flows, suspect this double-list skew.

**Evidence:** `src/topsport_agent/server/app.py::_wrap_with_extras`,
`_wrap_with_metrics` (both now also mutate `_capability_bundle`).
Regression captured by Langfuse API check: `plan-ctx-002` before the fix
had `observations=2` only for the parent trace; after the fix there are
separate traces `agent.run[<plan_id>:<step_id>:<hash>]` for every
sub-step.

---

## Declared sandbox field + unset-by-any-caller = no sandbox at all

**Context:** Reviewing multi-tenant exposure before enabling file_ops on HTTP.
`ToolContext.workspace_root: Path | None = None` was defined in
`types/tool.py`, and `tools/file_ops.py::_check_containment` implemented a
full resolve + is_relative_to + symlink-escape check that rejected any
path outside the root. Looked complete end-to-end in a code review.

Actually tested: a freshly booted server with `ENABLE_FILE_TOOLS=true`
let the LLM write `/etc/passwd_test_attack` unrestricted. Because
`engine/loop.py::_invoke_tool` built `ToolContext` with only
`session_id / call_id / cancel_event` — `workspace_root` stayed at its
default `None`, which `_check_containment` treats as CLI-trust-mode and
short-circuits the check. Type existed, logic existed, **no one ever
passed a value**.

**Learned:** This is the definitive "control plane exists, execution
plane doesn't wire it" pattern (codex flagged the category; this was a
live instance). The diagnostic signature:

1. `grep -rn "<field_name>" src/` shows **type definition** + **reader
   with conditional guard (`if x is None: return`)** — no writer.
2. No call site passes the field; callers construct the object with
   positional / partial kwargs only.
3. Tests exist but use direct object construction (bypass the broken
   pipeline) and happen to set the field when testing the sandbox logic,
   so the `None → skip` branch is never exercised in tests that matter.

Fix for this class of bug = follow the field back to source:

- Every type-level "sandbox" field needs a **default-producer** somewhere
  in the call chain (session creation → ToolContext construction →
  tool execution).
- If the tool runtime gets `None`, decide: is CLI-trust-mode the safe
  default, or should the tool fail closed? For workspace it's the former
  (back-compat with CLI); for `granted_permissions` it's the latter.
- Add a production-path test: boot the real server with defaults, try
  to escape, assert rejection. Not `write_file(ctx=FakeCtx(root=...))` —
  the bug was exactly in ctx construction.

Grep regex that finds this class of leak in CI:

```
field: <Type> | None = None          # definition
ctx.field is None                     # guard
```

If writer count is zero and tests don't cover the "server boot → invoke
tool" path, it's dead sandbox code.

**Evidence:** `src/topsport_agent/workspace/manager.py::WorkspaceRegistry`
(the missing writer + per-session directory allocator), now wired into
`server/app.py` via the SessionStore create hook. Regression test
`tests/test_workspace_sandbox.py::test_write_file_outside_workspace_rejected`
goes through the real tool pipeline, not just `_check_containment`
directly. E2E'd with `WORKSPACE_ROOT=/tmp/ts-ws-test`: attack path
`/etc/passwd_test_attack` was rejected; `alice` session's file written
inside `anonymous__ws-sess-alice/files/` succeeded.

---

## CapabilityModule: one shape for every Agent feature's install side-effects

**Context:** `Agent.from_config` was a god factory — each of 5 capabilities
(plugins / skills / memory / file_ops / browser) had its own hand-written
if-branch with a bespoke side-effect shape: skills added `tools +
context_providers`, plugins added `tools + event_subscribers + cleanup`,
browser added `tool_sources + cleanup`, etc. Adding image_gen or MCP or
tracing in a future PR meant editing the factory and remembering exactly
which 3-4 list mutations each capability needs. 8+ `enable_X` flags on
AgentConfig; the factory length grew linearly in number of capabilities.

**Learned:** The right primitive is "capability install has a uniform
output shape" — a `CapabilityBundle` that lists what the capability
contributes to each of the Agent's 6 extension axes (tools,
context_providers, tool_sources, post_step_hooks, event_subscribers,
cleanup_callbacks) plus a `state` dict for cross-capability publish/subscribe
(e.g. PluginsModule publishes `plugin_manager` → SkillsModule reads it to
discover skill_dirs). Every capability implements the same protocol:

```python
class CapabilityModule(Protocol):
    name: str
    def is_enabled(self, ctx: InstallContext) -> bool: ...
    def install(self, ctx: InstallContext) -> CapabilityBundle: ...
```

The factory becomes a 15-line loop: for each module, if enabled, install,
extend accumulator lists, merge state into `ctx.shared` for next module.
Agent construction is the same as before — only the assembly changed.

Two subtle wins that aren't obvious until you've migrated:

1. **Late binding via mutable reference cell**: PluginsModule's
   `spawn_agent` executor needs a reference to the fully-constructed Agent,
   but Agent isn't built until after all modules install. Pattern used:
   `parent_ref: list[Agent] = []` in InstallContext; PluginsModule captures
   it in a closure; factory appends `agent` after construction; closures
   dereference at tool-call time. Cleaner than post-construction patching.

2. **Ordering is simple linear, not DAG**: I briefly considered
   `depends_on` + topo-sort for capability ordering. With 5 modules the
   linearity is trivial (plugins → skills → rest); with 8+ a DAG matters.
   Ship the simpler form, upgrade only when ordering conflicts actually
   emerge. (Codex's critique was that the factory grows god-like; the
   protocol fixes the shape, the ordering issue is orthogonal and
   pre-extensible.)

Rule of thumb for this pattern: whenever you see N if-branches in a
constructor all mutating a shared "I'm accumulating state" pile, the
shared pile is the contract. Give it a name (`CapabilityBundle`), make
each branch return one, merge in a loop.

**Evidence:** `src/topsport_agent/agent/capabilities.py` (protocol) +
`capability_impls.py` (5 modules). Factory shrank from ~60 lines of
if/else to ~25 lines of module loop + accumulator merge. 913 tests
unchanged. E2E-verified: same server run exposes file_ops + memory +
skills + plugins simultaneously through the new assembly path.
