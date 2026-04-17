# Sub-Agent Spawn Contract

> How `spawn_agent` tool invokes a plugin agent as an isolated sub-Engine.

---

## Overview

Claude plugin agents (`agents/*.md` files in plugin packages) are exposed to
the LLM via `list_agents` and `spawn_agent` tools. `spawn_agent` creates an
isolated child Engine + Session that runs the agent's task, then returns the
final text to the parent.

Two modes exist:
1. **Metadata-only** (`executor=None`): returns agent definition without running.
   Used when no execution context is available (tests, preview).
2. **Real execution** (`executor=<callable>`): runs the child Engine and returns
   the final assistant text. This is what `Agent.from_config` wires up.

---

## Contracts

### SpawnExecutor type (`src/topsport_agent/plugins/agent_registry.py`)

```python
SpawnExecutor = Callable[
    ["AgentDefinition", str, ToolContext],
    Awaitable[dict[str, Any]],
]
```

- First arg: the `AgentDefinition` loaded from `agents/*.md`.
- Second arg: the task string (user-controlled input from LLM).
- Third arg: parent `ToolContext` (carries `session_id`, `call_id`, `cancel_event`).
- Returns: dict with at minimum `ok: bool`; `name` and `executed` are auto-filled
  by the spawn_agent wrapper if missing.

### Return shape conventions

Success:
```python
{
  "ok": True,
  "executed": True,
  "name": "plugin:agent",
  "text": "<final assistant output>",
  "tool_calls": <int>,        # count of tool calls made by sub-agent
  "messages": <int>,          # total messages in sub-session
}
```

Failure:
```python
{
  "ok": False,
  "executed": True,
  "name": "plugin:agent",
  "error": "<kind>: <message>",
  "partial_text": "<last assistant text if any>",
}
```

Not found (no execution attempted):
```python
{
  "ok": False,
  "error": "agent 'X' not found",
  "available": [<qualified names>],
}
```

---

## Default Executor (`src/topsport_agent/agent/base.py::_build_spawn_executor`)

```python
def _build_spawn_executor(
    provider: LLMProvider,
    parent_config: AgentConfig,
    get_parent_tools: Callable[[], list[ToolSpec]],
) -> SpawnExecutor:
```

### Model resolution

| AgentDefinition.model | Child Engine model |
| --- | --- |
| `"inherit"` | `parent_config.model` |
| Any other string (e.g. `"sonnet"`, `"opus"`) | passed through verbatim |

The string mapping is NOT interpreted by the framework — the value is handed
to the provider as-is. Providers are responsible for model name resolution.

### Tool filtering

```python
if agent.allowed_tools:          # non-empty list
    sub_tools = [t for t in parent_tools if t.name in set(agent.allowed_tools)]
else:                            # empty list or missing
    sub_tools = list(parent_tools)
```

- `allowed_tools=[]` (default) means inherit ALL parent tools.
- `allowed_tools=["Read", "Grep"]` restricts to named tools.
- No wildcard support.

### Delayed tool-set binding

`get_parent_tools` is a `lambda: list(tools)`, NOT a snapshot. This matters
because `Agent.from_config` registers `spawn_agent` BEFORE appending
skills/memory/browser tools. At executor invocation time, the lambda returns
the fully-populated tool list.

**If `get_parent_tools` snapshotted at build time, sub-agents would see only
the builtin and plugin-agent tools, missing skills/memory/file_ops.**

### Session isolation

```python
sub_session = Session(
    id=f"{ctx.session_id}:sub:{agent.qualified_name}:{uuid.uuid4().hex[:6]}",
    system_prompt=agent.body,
)
sub_session.messages.append(Message(role=Role.USER, content=task))
```

- Child session id includes parent id + agent name + random suffix for traceability.
- Child session_prompt is the agent's markdown body (NOT parent system prompt).
- No message history is shared with parent.

### Exception handling

```python
try:
    async for event in sub_engine.run(sub_session):
        ...
except Exception as exc:
    error = f"{type(exc).__name__}: {exc}"
```

Plus: `spawn_agent` wrapper in `build_agent_tools` wraps any exception from
the executor itself:
```python
try:
    result = await executor(agent, task, ctx)
except Exception as exc:
    return {"ok": False, "executed": True, "name": ..., "error": ...}
```

This two-level catch ensures sub-agent crashes NEVER propagate into the parent
Engine's ReAct loop.

---

## Unimplemented / Known Gaps

- **auto_skills activation**: `AgentDefinition.auto_skills` is parsed but NOT
  activated in the sub-engine. Child engines currently do not have their own
  `SkillMatcher`. Adding this requires sub-engine skill registry access.
- **Event propagation**: Child engine events are not forwarded to parent
  subscribers. Tracing/logging of sub-agent activity is lost.
- **Nested spawn**: Sub-agents technically inherit `spawn_agent` tool. There is
  no depth limit — runaway recursion relies on `max_steps` to terminate.

---

## Validation Matrix

| Input | executor | Expected |
| --- | --- | --- |
| Unknown agent name | any | `{ok: False, error: "not found"}` |
| Valid agent, executor=None | None | `{ok: True, executed: False, <metadata>}` |
| Valid agent, successful run | real | `{ok: True, executed: True, text: ...}` |
| Valid agent, model="inherit" | real | Sub engine uses `parent_config.model` |
| Valid agent, `allowed_tools=["X"]` | real | Sub engine tools = parent tools filtered by name |
| Valid agent, executor raises | real | `{ok: False, executed: True, error: "<type>: <msg>"}` |
| Valid agent, sub engine raises | real | `{ok: False, executed: True, error: ..., partial_text: ...}` |

---

## Forbidden Patterns

- DO NOT share `Session` instance between parent and child (breaks message
  invariants when both run concurrently).
- DO NOT bind `get_parent_tools` to a specific list at executor-build time;
  always use `lambda: list(tools)` for delayed eval.
- DO NOT re-raise executor exceptions; always convert to `{ok: False}` result.
- DO NOT forward `cancel_event` verbatim from parent to child; the child
  should have its own `cancel_event` (currently a minor gap — parent cancel
  does not propagate to sub-engine).

---

## Tests

Reference tests in `tests/test_spawn_agent.py`:
- `test_spawn_agent_without_executor_returns_metadata`: fallback mode
- `test_spawn_agent_delegates_to_executor`: happy path
- `test_spawn_agent_catches_executor_exception`: error isolation
- `test_real_executor_uses_inherit_model`: model resolution
- `test_real_executor_filters_allowed_tools`: tool filtering
- `test_real_executor_session_is_isolated`: session contract
- `test_default_agent_has_spawn_executor_wired`: integration with `Agent.from_config`
