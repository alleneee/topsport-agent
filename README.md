# topsport-agent

Agent runtime with a ReAct loop, multi-provider LLM abstraction, pluggable tools,
and session-scoped working memory.

## Status

Multi-agent plan mode landed. Browser control module added. 232 tests passing.

| Module | Location | State |
| --- | --- | --- |
| types | `src/topsport_agent/types/` | message, tool, session, events (incl. RUN_START/RUN_END) |
| llm | `src/topsport_agent/llm/` | `LLMRequest` / `LLMResponse` contract + provider Protocol |
| llm.clients | `src/topsport_agent/llm/clients/` | SDK client construction, env resolution, transport calls, transient retry |
| llm.providers | `src/topsport_agent/llm/providers/` | provider orchestration around SDK clients |
| llm.adapters | `src/topsport_agent/llm/adapters/` | provider-specific payload/response codecs |
| engine | `src/topsport_agent/engine/` | ReAct loop, cancel, hooks, event dispatch, planner, orchestrator |
| memory | `src/topsport_agent/memory/` | file store, injector, save/recall/forget tools |
| skills | `src/topsport_agent/skills/` | registry, loader, matcher, injector, load/unload/list tools |
| browser | `src/topsport_agent/browser/` | Playwright-based browser control with snapshot/ref interaction model |
| mcp | `src/topsport_agent/mcp/` | JSON config, lazy client, tool bridge, prompt/resource meta tools |
| tools | `src/topsport_agent/tools/` | executor (output cap + blob offload), safe_shell (execFile-only), blob store |
| observability | `src/topsport_agent/observability/` | Tracer alias, NoOpTracer, LangfuseTracer |
| cli | `src/topsport_agent/cli/` | interactive REPL, builtin tools (echo/calc/current_time) |
| tests | `tests/` | 232 passing |

## Quickstart

```bash
uv sync
uv run pytest -v
```

## CLI

Interactive REPL for verifying the engine end-to-end with a real LLM.

MODEL 格式为 `provider/model-name`，支持 `anthropic` 和 `openai` 两种 provider。
API 凭证使用通用的 `API_KEY` 和 `BASE_URL`，不带厂商前缀。

```bash
uv sync --group llm
```

通过 `.env` 文件配置（启动时自动加载）:

```env
API_KEY=sk-...
BASE_URL=https://api.example.com/v1
MODEL=anthropic/claude-sonnet-4-5
```

```bash
# 直接使用 .env 配置
uv run topsport-agent

# 命令行覆盖 model
uv run topsport-agent -m openai/gpt-4o
```

Built-in tools available in CLI mode: `echo`, `current_time`, `calc`.

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

## Browser Control

Install the optional dependency group:

```bash
uv sync --group browser
playwright install chromium
```

The browser module provides a `BrowserToolSource` that exposes 6 tools for web
page interaction, using a snapshot/ref model where the LLM references elements
by `@e1`, `@e2` etc. instead of CSS selectors.

```python
from topsport_agent.browser import BrowserClient, BrowserConfig, BrowserToolSource
from topsport_agent.engine import Engine, EngineConfig

browser_client = BrowserClient.from_config(BrowserConfig(headless=True))
browser_tools = BrowserToolSource(browser_client)

engine = Engine(
    provider,
    tools=[...],
    config=EngineConfig(model="claude-sonnet-4-5"),
    tool_sources=[browser_tools],
)

# After engine run completes:
await browser_client.close()
```

Available tools:

| Tool | Description |
| --- | --- |
| `browser_navigate` | Navigate to URL, return snapshot of interactive elements |
| `browser_snapshot` | Refresh the interactive element list with @refs |
| `browser_click` | Click by @ref or CSS selector, auto-snapshot on navigation |
| `browser_type` | Type text into input by @ref or CSS selector |
| `browser_screenshot` | Take screenshot, return file path |
| `browser_get_text` | Get text content from page or element |

Architecture follows the MCP module pattern: `BrowserClient` accepts an
injectable `page_factory` (like `MCPClient`'s `session_factory`), so all 34
tests run without Playwright installed. The browser is lazily initialized on
first use and scoped to the session lifetime.

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

## Multi-Agent Plan Mode

Plan mode lets an orchestrator agent generate a DAG of steps, pause for user
approval, then execute steps in parallel using isolated sub-agents.

### Usage

```python
from topsport_agent.engine import (
    EngineConfig, Orchestrator, Planner, SubAgentConfig,
)
from topsport_agent.types import StepDecision

# 1. Generate a plan
planner = Planner(provider, model="claude-sonnet-4-5")
plan = await planner.generate("Refactor auth module")

# 2. Review plan.steps — each has id, title, instructions, depends_on

# 3. Execute
config = SubAgentConfig(provider=provider, model="claude-sonnet-4-5", tools=[...])
orch = Orchestrator(plan, config, event_subscribers=[tracer])

async for event in orch.execute():
    if event.type == EventType.PLAN_WAITING:
        # Step failed — decide: retry / skip / abort
        orch.provide_decision(StepDecision.RETRY)
    print(event.type, event.payload)
```

### Architecture

- **Planner** sends a single LLM call with a `create_plan` tool. The LLM
  returns structured steps with dependency declarations.
- **Plan** validates the DAG on construction (Kahn's algorithm for cycle
  detection, dependency existence checks).
- **Orchestrator** executes the DAG by topological waves. Steps with all
  dependencies satisfied run in parallel via `asyncio.gather`. Each step gets
  a fully isolated `Engine` + `Session` instance.
- **Step configurators**: `StepConfigurator` hooks are called before each
  step. They can modify the sub-agent config (swap provider, inject tools,
  override model). Multiple configurators chain in order. A broken
  configurator is skipped with a warning.
- **Failure handling**: on step failure, `FailureHandler` hooks are tried
  first (auto-retry, auto-skip, auto-abort). If no handler is registered or
  all raise, the orchestrator emits `PLAN_WAITING` and pauses until
  `provide_decision()` is called. First handler to return wins.
- **Cancel propagation**: `Orchestrator.cancel()` sets its own cancel event
  and propagates to all running sub-engines.

### Plan Event Types

| Event | When |
| --- | --- |
| `plan.created` | Plan generated by planner |
| `plan.approved` | Orchestrator begins execution |
| `plan.step.start` | Sub-agent launched for a step |
| `plan.step.end` | Sub-agent completed (success or failure) |
| `plan.step.failed` | One or more steps in a wave failed |
| `plan.waiting` | Orchestrator paused, waiting for user decision |
| `plan.done` | All steps completed or skipped |
| `plan.failed` | Plan aborted or no ready steps remain |

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
- Plan DAG is validated on construction: no cycles, no self-deps, all dep IDs exist.
- Orchestrator cancel propagates to all running sub-engines.
- Orchestrator checks cancel immediately after `asyncio.gather` returns, before
  entering the failure-handling path.
- Skill activation is session-scoped — two sessions cannot see each other's active skills.
- Loading a skill via `load_skill` becomes visible on the next engine step without cache.
- The skill registry parses the real Claude Code `~/.claude/skills/` tree unchanged.
- MCP clients are lazy: no network or subprocess until the first list/call.
- MCP tool names are always prefixed with `<server>.` to prevent collisions.
- MCP tool errors and connection faults are reported as `is_error=True` in the
  result dict, not as engine-level exceptions.

## Not yet implemented

- HTTP or streaming surface for frontend integration
- Plan editing after approval
- Dynamic re-planning during execution
- Shared `common/frontmatter.py` -- memory, skills, and future modules currently
  carry their own copy
