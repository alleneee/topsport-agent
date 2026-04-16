# topsport-agent

Agent runtime with a ReAct loop, multi-provider LLM abstraction, pluggable tools,
and session-scoped working memory.

## Status

Phase 3 E — Loop detector, interject queue, concurrency guard landed. 143 tests passing.

| Module | Location | State |
| --- | --- | --- |
| types | `src/topsport_agent/types/` | message, tool, session, events (incl. RUN_START/RUN_END) |
| llm | `src/topsport_agent/llm/` | `LLMRequest` / `LLMResponse` contract + provider Protocol |
| llm.clients | `src/topsport_agent/llm/clients/` | SDK client construction, env resolution, transport calls, transient retry |
| llm.providers | `src/topsport_agent/llm/providers/` | provider orchestration around SDK clients |
| llm.adapters | `src/topsport_agent/llm/adapters/` | provider-specific payload/response codecs |
| engine | `src/topsport_agent/engine/` | ReAct loop, cancel, hooks, event dispatch, state machine |
| memory | `src/topsport_agent/memory/` | file store, injector, save/recall/forget tools |
| skills | `src/topsport_agent/skills/` | registry, loader, matcher, injector, load/unload/list tools |
| mcp | `src/topsport_agent/mcp/` | JSON config, lazy client, tool bridge, prompt/resource meta tools |
| tools | `src/topsport_agent/tools/` | executor (output cap + blob offload), safe_shell (execFile-only), blob store |
| observability | `src/topsport_agent/observability/` | Tracer alias, NoOpTracer, LangfuseTracer |
| tests | `tests/` | 115 passing (98 prior + 17 tools) |

## Quickstart

```bash
uv sync
uv run pytest -v
```

## 维护说明

- 核心运行链路、MCP 桥接、技能加载和文件记忆存储模块已补充中文代码注释，注释只解释流程意图和边界，不重复代码字面含义。

## Engine hooks

`Engine.__init__` accepts four optional hook collections:

- `context_providers` — return extra `Message` objects merged into the LLM call
  without being persisted into `session.messages`. Memory, skill, and MCP prompt
  injectors attach here.
- `tool_sources` — return dynamic `ToolSpec` lists merged into the per-step tool
  snapshot. MCP tool bridges attach here. Builtin tools win on name collision.
- `post_step_hooks` — async callbacks invoked after every step, including the
  final one. Memory writers and evaluators attach here.
- `event_subscribers` — receive every lifecycle `Event` (`RUN_START`, `STEP_START`,
  `LLM_CALL_*`, `TOOL_CALL_*`, `STATE_CHANGED`, `ERROR`, `CANCELLED`, `RUN_END`) in
  order. Exceptions in one subscriber do not affect the engine or other subscribers.
  Tracers and loggers attach here.

## Skills

Skills follow the Anthropic Agent Skills specification: each skill is a directory
containing a `SKILL.md` file with YAML-style frontmatter and a markdown body.

```markdown
---
name: echo-helper
description: Use when the user asks to echo back text.
version: 1.0.0
---

# Echo Helper

Just echo the input back to the user verbatim.
```

Required frontmatter keys: `name`, `description`. Any other key (e.g. `version`,
`argument-hint`, `license`) is preserved in `SkillManifest.extra`.

Activation model — metadata always resident, full body loaded on demand:

- `SkillInjector` always renders a catalog of available skills (name + description)
  into the system prompt, so the LLM knows what exists without paying the token cost
  of every body.
- The LLM calls the `load_skill` tool to activate a skill for the current session.
  From the next step onward, `SkillInjector` includes the activated skill's full body.
- `unload_skill` deactivates it; `list_skills` inspects the catalog with active state.

Session-scoped: activation state lives in `SkillMatcher`, keyed by `session.id`.
Two concurrent sessions cannot interfere with each other's active skill set.

The registry parses the real `~/.claude/skills/` tree as part of the test suite —
any Claude skill that works with Claude Code's loader works here too.

## LLM Providers — Anthropic

Install the optional dependency group:

```bash
uv sync --group llm
```

```python
from topsport_agent.engine import Engine, EngineConfig
from topsport_agent.llm.providers import AnthropicProvider

provider = AnthropicProvider(
    max_tokens=4096,
    thinking_budget=2048,
)

engine = Engine(
    provider,
    tools=[...],
    config=EngineConfig(model="claude-sonnet-4-5"),
)
```

Runtime-to-Anthropic translation contract:

| Runtime | Anthropic API |
| --- | --- |
| `Role.SYSTEM` messages | Collapsed into the top-level `system` parameter (`\n\n` joined) |
| `Role.USER` message | `{"role": "user", "content": "<text>"}` |
| `Role.ASSISTANT` with text + tool calls | `{"role": "assistant", "content": [TextBlock, ToolUseBlock, ...]}` |
| Consecutive `Role.TOOL` messages | Merged into a single `{"role": "user", "content": [ToolResultBlock, ...]}` |
| `ToolSpec` | `{"name", "description", "input_schema"}` |
| `ToolResult.is_error=True` | `{"type": "tool_result", "is_error": True, ...}` |
| `LLMRequest.provider_options["anthropic"]["thinking"]` or constructor `thinking_budget` | Top-level `thinking` param |

`AnthropicProvider` now only orchestrates request flow. `AnthropicMessagesClient`
owns SDK client construction, env resolution, and the transport call, while
`AnthropicMessagesAdapter` handles request/response translation. The provider is
constructor-injectable (`AnthropicProvider(client=...)`) so tests run against a
mock client without the real `anthropic` package. `ANTHROPIC_API_KEY` and
`ANTHROPIC_BASE_URL` environment variables are picked up automatically.
Provider-native assistant payload is preserved under
`Message.extra["llm_response"]`, as the serialized form of
`ProviderResponseMetadata`. Its `assistant_blocks` field follows a typed block
shape (`text` / `thinking` / `tool_use`), similar to AgentScope's content block
design, so downstream code can inspect Anthropic content blocks without
rebuilding them from `text` and `tool_calls`.

## LLM Providers — OpenAI

```python
from topsport_agent.llm.providers import OpenAIChatProvider

provider = OpenAIChatProvider(
    max_tokens=4096,
    reasoning_effort="medium",  # "low" | "medium" | "high" | None
)
```

Runtime-to-OpenAI chat completions translation contract:

| Runtime | OpenAI chat completions |
| --- | --- |
| `Role.SYSTEM` message | Stays as `{"role": "system", "content": ...}` — multiple system messages are preserved in order |
| `Role.USER` message | `{"role": "user", "content": "<text>"}` |
| `Role.ASSISTANT` with text | `{"role": "assistant", "content": "<text>"}` |
| `Role.ASSISTANT` with `tool_calls` | `{"role": "assistant", "content": null, "tool_calls": [{id, type: "function", function: {name, arguments: <JSON str>}}]}` |
| Each `Role.TOOL` message | Independent `{"role": "tool", "tool_call_id": id, "content": <str>}` — one per tool result |
| `ToolSpec` | `{"type": "function", "function": {name, description, parameters}}` |
| `LLMRequest.provider_options["openai"]["reasoning_effort"]` or constructor `reasoning_effort` | Top-level `reasoning_effort` param for o-series / reasoning models |
| `LLMRequest.provider_options["openai"]["max_completion_tokens"]` | Replaces `max_tokens` (required for reasoning models) |

Key contrasts with Anthropic:

- OpenAI tool call `arguments` is a **JSON string**, not a dict — adapter does
  `json.dumps` on the way out and `json.loads` on the way back.
- System messages are **not** lifted to a top-level parameter; they stay in the
  messages array.
- Each `ToolResult` stays as its own `role=tool` message; no merging into user
  content blocks.
- Malformed tool argument JSON is captured as `{"_raw_arguments": "<text>"}`
  instead of raising — keeps the loop alive when the model emits broken JSON.
- Usage fields `prompt_tokens` / `completion_tokens` are normalized to
  `input_tokens` / `output_tokens` to match the Anthropic adapter's output.

`OpenAIChatProvider` now only orchestrates request flow. `OpenAIChatClient`
owns SDK client construction, env resolution, and the transport call, while
`OpenAIChatAdapter` handles request/response translation. Environment variables
picked up automatically: `OPENAI_API_KEY`, `OPENAI_BASE_URL`,
`OPENAI_ORGANIZATION`. Constructor-injectable for testing.
Provider-native assistant payload is preserved under
`Message.extra["llm_response"]`, as the serialized form of
`ProviderResponseMetadata`. Its `assistant_blocks` field follows a typed block
shape (`text` / `thinking` / `tool_use`), similar to AgentScope's content block
design, so downstream code can inspect OpenAI assistant output without
polluting the root of `Message.extra`.

## MCP (Model Context Protocol)

Install the optional dependency group:

```bash
uv sync --group mcp
```

MCP servers are configured via a JSON file compatible with the Claude Desktop
convention:

```json
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/path/to/dir"],
      "env": {}
    },
    "remote-api": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer token"},
      "timeout": 30
    }
  }
}
```

Load it and wire into the engine:

```python
from pathlib import Path
from topsport_agent.engine import Engine, EngineConfig
from topsport_agent.mcp import MCPManager, build_mcp_meta_tools

mcp = MCPManager.from_config_file(Path("mcp.json"))

engine = Engine(
    provider,
    tools=build_mcp_meta_tools(mcp),
    config=EngineConfig(model="..."),
    tool_sources=mcp.tool_sources(),
)
```

What the bridge exposes:

- **MCP tools** → auto-merged into the per-step tool snapshot as
  `<server>.<tool>` names via `MCPToolSource` (a `ToolSource` implementation).
- **MCP prompts** → accessed via meta-tools `list_mcp_prompts` and
  `get_mcp_prompt`. The agent can discover and render prompts on demand.
- **MCP resources** → accessed via meta-tools `list_mcp_resources` and
  `read_mcp_resource`. Returns text content from the MCP server.

Lifecycle:

- **Lazy connect**: clients do not connect until the first `list_tools` /
  `list_prompts` / `list_resources` / `call_tool` / `get_prompt` / `read_resource` call.
- **Per-call session for calls**: `call_tool`, `get_prompt`, and `read_resource`
  open a fresh session each invocation — safe across tasks, safe on cancel.
- **Cached lists**: `list_tools` / `list_prompts` / `list_resources` cache the
  first response. Use `force_refresh=True` or `invalidate_cache()` to refetch.
- **Testable in isolation**: `MCPClient(name, session_factory)` takes any async
  context manager that yields a session-shaped object, so tests run without the
  real `mcp` package installed.

## Tracing with Langfuse

Install the optional dependency group:

```bash
uv sync --group tracing
```

Wire `LangfuseTracer` as an event subscriber:

```python
from topsport_agent.engine import Engine, EngineConfig
from topsport_agent.observability import LangfuseTracer

tracer = LangfuseTracer()  # reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL

engine = Engine(
    provider,
    tools=[...],
    config=EngineConfig(model="claude-sonnet-4-5"),
    event_subscribers=[tracer],
)
```

Mapping of runtime events to Langfuse observations:

| Event | Langfuse observation |
| --- | --- |
| `RUN_START` | root `agent` observation, trace `session_id` set |
| `STEP_START` / `STEP_END` | child `span` per step |
| `LLM_CALL_START` / `LLM_CALL_END` | `generation` with `model`, `usage_details` |
| `TOOL_CALL_START` / `TOOL_CALL_END` | `tool` observation, `level=ERROR` on failure |
| `ERROR` | root `level=ERROR` + `status_message` |
| `CANCELLED` | root `level=WARNING` + `status_message=cancelled` |
| `RUN_END` | root `end()` + `flush()` |

The tracer is constructor-injectable (`LangfuseTracer(client=...)`) so tests can
drive it with a mock client — no real Langfuse dependency required for testing.

## Working memory

Session-scoped. Each entry is a markdown file with frontmatter:

```markdown
---
name: Task goal
description: What the user wants done
type: goal
key: task_goal
created_at: 2026-04-15T14:22:00
updated_at: 2026-04-15T14:22:00
---

Refactor ingest pipeline to use async generators.
```

Supported types: `goal`, `identity`, `fact`, `constraint`, `note`.

Default storage: `FileMemoryStore(base_path)` → `<base_path>/<session_id>/<slug>.md`.
Swap implementations by writing any class that satisfies the `MemoryStore` Protocol.

`MemoryInjector` is a `ContextProvider` — attach it to `Engine.context_providers`
and working memory is injected as a `system` message every step.

`build_memory_tools(store)` returns `save_memory`, `recall_memory`, `forget_memory`
tool specs the agent can call directly.

## Verified invariants

- `tool_calls` are always followed immediately by matching `tool_result` messages.
- `Engine.cancel()` preempts an in-flight LLM call through `asyncio.wait(FIRST_COMPLETED)`.
- Unknown tools produce an error `ToolResult` instead of crashing the loop.
- `max_steps` terminates cleanly into `RunState.DONE`.
- Tool list is snapshotted per step and merged with `tool_sources`, so hot-registered
  tools are visible next turn.
- Ephemeral context messages from providers never leak into `session.messages`.
- `session.system_prompt` plus ephemeral `SYSTEM` messages collapse into one system
  block per LLM call.
- Builtin tool names win over tool source names on collision.
- Every `engine.run()` is bookended by `RUN_START` and `RUN_END` events.
- Event subscribers receive events in the same order they are yielded.
- A subscriber raising an exception does not break the engine or other subscribers.
- Skill activation is session-scoped — two sessions cannot see each other's active skills.
- Loading a skill via `load_skill` becomes visible on the next engine step without cache.
- The skill registry parses the real Claude Code `~/.claude/skills/` tree unchanged.
- MCP clients are lazy: no network or subprocess until the first list/call.
- MCP tool names are always prefixed with `<server>.` to prevent collisions.
- MCP tool errors and connection faults are reported as `is_error=True` in the
  result dict, not as engine-level exceptions.

## Not yet implemented

- Tool executor with output caps and blob offload (Phase 3 C)
- Context compaction (micro, auto, goal reinjection) (Phase 3 D)
- Loop detector and interject-message queue (Phase 3 E)
- Concurrency lock with take-and-release semantics (Phase 3 E)
- HTTP or streaming surface for frontend integration (Phase 3 F)
- Shared `common/frontmatter.py` — memory, skills, and future modules currently
  carry their own copy; will be unified in Phase 3 G polish
