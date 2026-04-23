# topsport-agent

Agent runtime with a ReAct loop, multi-provider LLM abstraction, pluggable tools,
and session-scoped working memory.

## Status

Multi-agent plan mode landed. Browser control module added. Claude plugin
ecosystem integration added. Tagged prompt section system added. Agent
abstraction layer with default/browser presets. File operation tool suite.
Real sub-agent execution via spawn_agent. LLM streaming output. Conditional
plan DAG with shared `PlanContext`, post-condition refine loops, and
plan-level checkpointing. Claude-Code schema parity: typed event payloads,
extended `ToolSpec` metadata, automatic blob offload for large tool outputs,
pydantic-driven input schemas, and an injectable permission system
(checker + asker, fail-closed). 705 tests passing.

| Module | Location | State |
| --- | --- | --- |
| types | `src/topsport_agent/types/` | message, tool (+metadata / `from_model`), session, events, event_payloads, plan, plan_context, permission |
| llm | `src/topsport_agent/llm/` | `LLMRequest` / `LLMResponse` contract + provider Protocol |
| llm.clients | `src/topsport_agent/llm/clients/` | SDK client construction, env resolution, transport calls, transient retry |
| llm.providers | `src/topsport_agent/llm/providers/` | provider orchestration around SDK clients |
| llm.adapters | `src/topsport_agent/llm/adapters/` | provider-specific payload/response codecs |
| engine | `src/topsport_agent/engine/` | ReAct loop, cancel, hooks, planner, orchestrator, checkpoint, plan_context_tools, permission |
| memory | `src/topsport_agent/memory/` | file store, injector, save/recall/forget tools |
| skills | `src/topsport_agent/skills/` | registry, loader, matcher, injector, load/unload/list tools |
| browser | `src/topsport_agent/browser/` | Playwright-based browser control with snapshot/ref interaction model |
| mcp | `src/topsport_agent/mcp/` | JSON config, lazy client, tool bridge, prompt/resource meta tools |
| tools | `src/topsport_agent/tools/` | executor (output cap + blob offload), safe_shell (execFile-only), blob store |
| observability | `src/topsport_agent/observability/` | Tracer alias, NoOpTracer, LangfuseTracer |
| plugins | `src/topsport_agent/plugins/` | Claude Code plugin ecosystem: discovery, skills, agents, hooks |
| agent | `src/topsport_agent/agent/` | high-level Agent abstraction with default/browser presets |
| cli | `src/topsport_agent/cli/` | interactive REPL, builtin tools (echo/calc/current_time) |
| tests | `tests/` | 705 passing |

## Quickstart

```bash
uv sync
uv run pytest -v
```

## Agent Abstraction

The `Agent` class packages `Engine` + skills + memory + plugins + browser
into a single high-level object. Two preset factories ship out of the box:

```python
from topsport_agent import default_agent, browser_agent

# General-purpose agent with all standard capabilities
agent = default_agent(provider, model="anthropic/claude-sonnet-4-5")

# Browser automation specialist (requires playwright)
agent = browser_agent(provider, model="anthropic/claude-sonnet-4-5")

session = agent.new_session()
async for event in agent.run("navigate to example.com and extract the title", session):
    ...
await agent.close()
```

### AgentConfig Capability Switches

| Field | Default | Effect when enabled |
| --- | --- | --- |
| `enable_skills` | `True` | Load skills from `local_skill_dirs` + plugin skill dirs; mount `load_skill`/`unload_skill`/`list_skills` tools |
| `enable_memory` | `True` | `FileMemoryStore` under `memory_base_path`; mount `save_memory`/`recall_memory`/`forget_memory` |
| `enable_plugins` | `True` | Full Claude plugin ecosystem loaded; expose `list_agents`/`spawn_agent`; hooks wired |
| `enable_browser` | `False` | Playwright-backed browser control with 10 `browser_*` tools (snapshot/ref model + iframe scope) |

### Available Presets

- **`default_agent(provider, model, ...)`**: all capabilities on,
  `DEFAULT_SYSTEM_PROMPT` describes the full toolset
- **`browser_agent(provider, model, ...)`**: browser mandatory (raises
  `BrowserUnavailableError` if Playwright missing); specialized prompt
  explains `@ref` snapshot model and browser workflow

Custom Agents are built via `Agent.from_config(provider, AgentConfig(...))`
— declare the capability flags and extras you need.


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

## HTTP Server

OpenAI-compatible chat completions endpoint plus a plan execution endpoint,
both streaming via Server-Sent Events. Stateful sessions, per-session LRU,
client-disconnect cancellation.

```bash
uv sync --group llm --group api
```

Reuses the CLI `.env` (`API_KEY` / `BASE_URL` / `MODEL`). Optional:
`HOST`, `PORT`, `SESSION_TTL_SECONDS`, `MAX_SESSIONS`.

```bash
uv run topsport-agent-serve --host 0.0.0.0 --port 8000
```

### POST /v1/chat/completions

OpenAI wire format. The `user` field is repurposed as `session_id`
(auto-generated when omitted). Server-side history is authoritative: only
the last `role=user` message in the request body is taken as new input.

```bash
# non-streaming JSON
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "anthropic/claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "user": "my-session"
    }'

# streaming SSE
curl -N http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "anthropic/claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    }'
```

SSE frames are standard OpenAI `chat.completion.chunk` objects terminated
by `data: [DONE]`. Only `delta.content` and `finish_reason` are emitted —
internal tool-call events are not surfaced.

### POST /v1/plan/execute

Run a DAG of sub-agent steps, each with its own isolated Engine+Session,
streamed as named SSE events.

```bash
curl -N http://localhost:8000/v1/plan/execute \
    -H "Content-Type: application/json" \
    -d '{
        "model": "anthropic/claude-sonnet-4-5",
        "plan": {
            "id": "p1",
            "goal": "fan-out",
            "steps": [
                {"id": "s1", "title": "A", "instructions": "Answer: ALPHA"},
                {"id": "s2", "title": "B", "instructions": "Answer: BETA"},
                {"id": "s3", "title": "sum",
                 "instructions": "Say DONE",
                 "depends_on": ["s1","s2"]}
            ]
        }
    }'
```

Event names: `plan_approved`, `plan_step_start`, `plan_step_end`,
`plan_step_failed`, `plan_waiting`, `plan_done`, `plan_failed`,
`cancelled`, `error`. Failed steps with no human-in-the-loop adjudicator
are auto-aborted (`PLAN_WAITING` → `ABORT`) to avoid hangs.

### Session semantics

- `user` → `session_id`; first request creates an Agent, subsequent requests reuse it.
- Per-session `asyncio.Lock` serializes concurrent requests on the same id.
- LRU eviction at `MAX_SESSIONS`; TTL sweep at `SESSION_TTL_SECONDS` (default 3600).
- Eviction calls `agent.close()` (cleans up plugins / browser if enabled).
- Client disconnect triggers `agent.cancel()` / `orchestrator.cancel()`.

## 维护说明

- 核心运行链路、MCP 桥接、技能加载和文件记忆存储模块已补充中文代码注释，注释只解释流程意图和边界，不重复代码字面含义。

## Prompt Section System

System prompts are organized into tagged sections, inspired by Claude Code's
XML tag structure. Each section has a tag name and priority (lower = earlier).

```
<system-prompt>
You are a helpful assistant.
</system-prompt>

<working-memory>
[goal] Refactor pipeline — current task
</working-memory>

<skills-catalog>
- `superpowers:tdd`: Use when implementing features...
</skills-catalog>

<active-skills>
## Skill: superpowers:tdd
[full skill body]
</active-skills>
```

### Built-in Section Priorities

| Priority | Tag | Source |
| --- | --- | --- |
| 0 | `system-prompt` | `session.system_prompt` |
| 100 | `identity` | Agent identity (future) |
| 200 | `working-memory` | `MemoryInjector` |
| 300 | `skills-catalog` | `SkillInjector` (catalog) |
| 400 | `active-skills` | `SkillInjector` (active bodies) |
| 500 | `plugin-context` | Plugin context (future) |
| 600 | `tools-guide` | Tool guidance (future) |
| 700 | `session-state` | Session state (future) |
| 900 | `instructions` | High-priority instructions |

### Selective Compaction

The tag structure enables targeted context compression:

- **Protected tags** (`system-prompt`, `identity`, `instructions`): never
  dropped during compaction
- **Compressible tags** (`session-state`, `tools-guide`, `plugin-context`,
  `skills-catalog`, `working-memory`): dropped in order when token budget
  is exceeded
- `PromptBuilder.build_with_budget()` automatically drops compressible
  sections to fit within a token limit
- `compact_system_prompt(text, drop_tags=...)` removes specific tagged
  sections from an assembled prompt
- `extract_protected_sections(text)` keeps only protected sections for
  extreme compression scenarios

### Custom Sections

ContextProviders declare sections via `Message.extra`:

```python
Message(
    role=Role.SYSTEM,
    content="my context",
    extra={"section_tag": "my-custom-section", "section_priority": 550},
)
```

Providers without section metadata default to tag `"context"`, priority 500.

## Engine hooks

`Engine.__init__` accepts these optional injection points:

### Capability hooks

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

### Safety / resource hooks

- `sanitizer` — prompt-injection defense applied to `untrusted` tool results
  before they land in `session.messages`.
- `blob_store` + `default_max_result_chars` — automatic offload of oversized tool
  outputs: full payload goes to blob storage, preview + `blob_ref` returned to
  LLM. See [Tool Metadata & Concurrency](#tool-metadata--concurrency).
- `permission_checker` + `permission_asker` — gate tool handler execution
  behind a two-stage ALLOW/DENY/ASK decision. See
  [Permission System](#permission-system).

## Typed Event Payloads

Each `EventType` has a declared pydantic payload schema in
`types/event_payloads.py`. Subscribers keep the existing
`event.payload.get(...)` dict access for back-compat; new code can call
`event.typed_payload()` for a validated, strongly-typed model instance:

```python
async def on_event(self, event: Event) -> None:
    if event.type == EventType.LLM_CALL_END:
        # dict access still works
        tokens = event.payload.get("usage", {})
        # typed access gives IDE autocomplete and runtime validation
        typed = event.typed_payload()  # LLMCallEndPayload
        assert typed.tool_call_count >= 0
```

Design choices (parallel schema, not replacement):

- `Event.payload: dict[str, Any]` is untouched — 20+ existing consumers (CLI,
  server SSE, tracer, metrics, hook runner) keep working.
- Each payload model uses `extra="ignore"` so publishers can add debug fields
  without breaking subscribers. The schema is the **minimum contract** for
  subscribers, not the maximum contract for publishers.
- `frozen=True` prevents one subscriber from mutating the model before the
  next subscriber sees it.
- Module load asserts every `EventType` has a registered schema — forgetting
  to register a new event fails fast.

## Tool Metadata & Concurrency

`ToolSpec` carries declarative metadata that drives permission, concurrency,
and result-size policies:

| Field | Default | Purpose |
| --- | --- | --- |
| `trust_level` | `"trusted"` | `"untrusted"` routes through `sanitizer` before landing in session |
| `read_only` | `False` | No side effects (pure query); default permission allow |
| `destructive` | `False` | Irreversible (delete/overwrite/send); default permission asks |
| `concurrency_safe` | `False` | Engine may run with other safe tools via `asyncio.create_task` |
| `max_result_chars` | `None` | Trigger blob offload when result exceeds cap |
| `validate_input` | `None` | Async pre-flight; returning a string skips handler and sends the error to the LLM |
| `input_schema` | `None` | Pydantic `BaseModel` class; `ToolSpec.from_model(...)` auto-exports `parameters` |

### Concurrent tool execution

When an assistant turn produces multiple `tool_calls`, the engine pre-schedules
every `concurrency_safe` handler via `asyncio.create_task` while still yielding
`TOOL_CALL_START/END` events in the original call order. Unsafe tools run
synchronously in their turn. Observability is preserved; wall-clock wins are
real (three 50 ms safe calls → ~55 ms total instead of 150 ms).

### Pydantic input schemas

```python
from pydantic import BaseModel, Field
from topsport_agent.types.tool import ToolSpec, ToolContext

class SearchInput(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)

async def search(inp: SearchInput, ctx: ToolContext) -> list[str]:
    return do_search(inp.query, inp.limit)

spec = ToolSpec.from_model(
    name="search",
    description="Search the knowledge base",
    input_model=SearchInput,
    handler=search,          # typed handler signature
    read_only=True,
    concurrency_safe=True,
)
# spec.parameters is auto-generated from SearchInput.model_json_schema()
```

Validation errors are caught automatically and returned as
`{"error": "invalid_input", "detail": [...]}` so the LLM can self-correct.

### Blob offload

```python
from topsport_agent.tools.blob_store import FileBlobStore
from topsport_agent.engine.loop import Engine, EngineConfig

engine = Engine(
    provider, tools, EngineConfig(model="..."),
    blob_store=FileBlobStore("/tmp/blobs"),
    default_max_result_chars=20_000,  # per-tool override via ToolSpec.max_result_chars
)
```

When a tool returns more than the cap, the LLM sees
`{"truncated": True, "original_size": N, "cap": C, "blob_ref": "blob://...", "preview": "..."}`
while the full payload lives on disk.

## Permission System

Two-stage injectable decision flow modeled after the Claude-Code permission
layer, minus the persisted rule set (that's a separate concern for deployment
configuration):

```python
from topsport_agent.engine.permission import (
    DefaultPermissionChecker,
    AlwaysDenyAsker,
)

engine = Engine(
    provider, tools, EngineConfig(model="..."),
    permission_checker=DefaultPermissionChecker(),
    permission_asker=AlwaysDenyAsker(),  # or your interactive CLI/server asker
)
```

### Decision flow

1. `permission_checker.check(tool, call, ctx)` returns `ALLOW` / `DENY` / `ASK`.
2. On `ASK`, `permission_asker.ask(tool, call, ctx, reason)` gives the final
   `ALLOW` / `DENY`.
3. `ALLOW` may include `updated_input` — the handler receives the rewritten
   arguments instead of the LLM's original (path normalization, safety
   injection, etc.).
4. `DENY` short-circuits the handler; the reason becomes the `ToolResult`
   error content the LLM sees.

### Fail-closed defaults

Every non-ALLOW branch denies when something goes wrong:

| Failure | Behavior |
| --- | --- |
| `permission_checker=None` | No checks (opt-in API; back-compat) |
| Checker raises | `DENY` with `"checker error: ..."` |
| Checker returns `ASK`, no asker configured | `DENY` |
| Asker raises | `DENY` with `"asker error: ..."` |
| Asker returns `ASK` (contract violation) | `DENY` |

`PermissionDecision` is `frozen=True` so subscribers cannot mutate a decision
between the checker and the handler.

### Default checker policy

`DefaultPermissionChecker` consumes `ToolSpec` metadata:

- `destructive=True` → `ASK` (require asker approval)
- `read_only=True` → `ALLOW`
- Otherwise → `ALLOW` (back-compat with tools that predate metadata)

Custom policies (tenant blacklists, regex-on-args, cached "always allow this
command" state) implement the `PermissionChecker` Protocol directly.

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

The browser module provides a `BrowserToolSource` that exposes 10 tools for web
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
| `browser_back` | Go back one step in history and return a fresh snapshot |
| `browser_snapshot` | Refresh the @ref list; pass `frame_selector` to scope into an iframe |
| `browser_click` | Click by @ref or CSS selector, auto-snapshot on navigation |
| `browser_type` | Type text into input by @ref or CSS selector |
| `browser_press_key` | Dispatch a key / chord (`Enter`, `Control+A`) at page or element level |
| `browser_select_option` | Select one or more options in a `<select>` element |
| `browser_wait_for` | Wait for selector state (visible/hidden/attached/detached) and/or sleep |
| `browser_screenshot` | Take screenshot, return file path |
| `browser_get_text` | Get text content from page or element |

**Ref disambiguation**: when the same `(role, name)` repeats on one page (e.g.
multiple "Save" buttons), entries are auto-assigned `nth=0,1,2...` and the
client resolves via `get_by_role(...).nth(n)`; unique elements keep `nth=None`
and retain the `.first` fast-path.

**iframe scope**: `browser_snapshot(frame_selector="iframe#main")` switches the
snapshot scope into that iframe and marks it sticky — subsequent `click` /
`type` / `get_text` / `press_key` / `select_option` inherit the frame until the
next navigation (explicit `navigate` / `back`, or a `click` that triggers one)
automatically clears it.

**URL policy**: `BrowserURLPolicy` (default-on) blocks non-http(s) schemes,
loopback, RFC1918 private ranges, link-local, and known cloud metadata
endpoints (`169.254.169.254`, `metadata.google.internal`, `100.100.100.200`).
Pass `url_policy=BrowserURLPolicy(allow_private=True, ...)` to loosen for
controlled environments; loopback and metadata remain blocked.

Architecture follows the MCP module pattern: `BrowserClient` accepts an
injectable `page_factory` (like `MCPClient`'s `session_factory`), so the full
75-test browser suite runs without Playwright installed. The browser is lazily
initialized on first use and scoped to the session lifetime.

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

## Structured Logging

Stdlib `logging` output can be switched to single-line JSON for log aggregation
platforms (ELK, Loki, Datadog). No extra dependency.

Enable at the server boot:

```bash
LOG_FORMAT=json LOG_LEVEL=INFO uv run topsport-agent-serve
```

Or programmatically:

```python
import logging
from topsport_agent.observability.logging import configure_json_logging

configure_json_logging(level=logging.INFO)

_logger = logging.getLogger("topsport_agent.app")
_logger.warning(
    "session closed",
    extra={"session_id": sid, "tenant_id": tid, "principal": user},
)
```

Output:

```json
{"ts":"2026-04-22T09:18:22+00:00","level":"WARNING","logger":"topsport_agent.app","msg":"session closed","session_id":"sess-abc","tenant_id":"t1","principal":"niko"}
```

Reserved fields: `ts`, `level`, `logger`, `msg`. Any keys passed via `extra={}`
are merged alongside the reserved fields. Exceptions render as `exc` with the
full traceback string.

The hot server paths (`chat_stream_failed`, `plan_stream_failed`,
`session_create_hook_failed`, `session_close_hook_failed`, `plan_execute`) emit
`event` + `session_id` / `tenant_id` / `principal` / `plan_id` via `extra={}`
so log filters can scope by tenant or session without regex parsing.

`configure_json_logging` is idempotent — repeated calls do not stack handlers.

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
| `plan.step.start` | Sub-agent launched for a step (payload includes `iteration`) |
| `plan.step.end` | Sub-agent completed (success or failure); payload includes `iterations` |
| `plan.step.skipped` | Step pre-condition returned False or raised |
| `plan.step.loop` | Post-condition returned False; step looped back, downstream reset |
| `plan.step.failed` | One or more steps in a wave failed |
| `plan.waiting` | Orchestrator paused, waiting for user decision |
| `plan.done` | All steps completed or skipped |
| `plan.failed` | Plan aborted or no ready steps remain |

### Conditional DAG — `condition`, `post_condition`, `max_iterations`

`PlanStep` accepts two callables that read a shared `PlanContext` and a
bounded loop counter:

- `condition(ctx) -> bool` — evaluated at the start of every wave for each
  ready step. Returning `False` marks the step `SKIPPED` and emits
  `plan.step.skipped`. Raising is treated as "not satisfied" and recorded in
  `step.error`.
- `post_condition(ctx) -> bool` — evaluated after a step's sub-agent
  finishes. Returning `False` sends the step back to `PENDING` and resets all
  transitive downstream steps via `Plan.reset_dependents_of()` (DONE/FAILED
  → PENDING, RUNNING/SKIPPED preserved, `iterations` counter preserved so
  `max_iterations` is a hard ceiling). Exhausting `max_iterations` flips the
  step to `FAILED` and enters the normal failure-handler flow.
- `max_iterations: int = 1` — default is 1 (no looping). Set higher to
  enable reflect-revise style loops with a guaranteed ceiling.

Evaluation timing: both conditions run on an **immutable context snapshot
taken at wave start**, so steps in the same wave see the same context and
their order of completion does not affect each other's gating.

### Shared State — `PlanContext`

`PlanContext` is a pydantic `BaseModel` subclass users extend to declare the
shared fields a plan needs. Merge semantics are declared per-field via
`Annotated[..., Reducer(fn)]`:

```python
from typing import Annotated
import operator
from topsport_agent.types.plan_context import PlanContext, Reducer

class MyCtx(PlanContext):
    findings: Annotated[list[str], Reducer(operator.add)] = []  # append
    attempts: Annotated[int, Reducer(lambda a, b: a + b)] = 0   # sum
    mode: str = "exploring"                                      # override
```

Rules enforced at construction time:

- `extra="forbid"` — typos in field names raise immediately; LLM-driven
  writes cannot silently create junk fields.
- `validate_assignment=True` — `Field(ge=..., le=...)` constraints still fire
  on merge, so invalid values are rejected at write time.
- Declaring two `Reducer` instances on one field raises `ValueError` at the
  first merge — forces ambiguous merge semantics to be written as one
  function instead of being implicit.

### Sub-agent Tools: `plan_context_read` / `plan_context_merge`

When a `Plan` is constructed with a `context=` value, the orchestrator
auto-mounts two tools onto every sub-agent's Engine via a
`PlanContextToolSource`:

- `plan_context_read()` — returns the current context snapshot (JSON).
- `plan_context_merge(key, value)` — applies the declared reducer for `key`
  (or overwrites if none) and returns the new snapshot.

Concurrency: all merges funnel through an `asyncio.Lock` inside
`PlanContextBridge`, so concurrent sub-agents in the same wave cannot lose
updates.

Omitting `context` (`Plan(..., context=None)`) leaves these tools
unregistered, so sub-agents of context-less plans have a clean tool surface.

### Checkpointing

Every plan-level event boundary (APPROVED, end-of-wave, SKIPPED, LOOP,
terminal) writes a `PlanSnapshot` via the configured `Checkpointer`:

```python
from topsport_agent.engine.checkpoint import FileCheckpointer, PlanSnapshot

ckpt = FileCheckpointer("./checkpoints")
orch = Orchestrator(plan, config, checkpointer=ckpt)
async for event in orch.execute():
    ...

# Crash recovery — build the same Plan skeleton, then apply saved state:
plan2 = build_same_plan_skeleton()
snap = await ckpt.load("review-job-42")
if snap:
    snap.apply_to(plan2, context_cls=MyCtx)
orch2 = Orchestrator(plan2, config, checkpointer=ckpt)
```

Snapshots carry only mutable state (`status`, `result`, `error`,
`iterations`, `context_data`) — `condition`, `post_condition`, and
`depends_on` are code-level and must be rebuilt by the caller. Built-in
backends:

- `MemoryCheckpointer` — process-local dict (testing, single-run).
- `FileCheckpointer(base_dir)` — one JSON per `plan_id`, atomic `tmp + rename`.
  Rejects plan ids containing `/`, `..`, or NUL.

Custom backends implement the `Checkpointer` Protocol
(`async save(snapshot) -> None` / `async load(plan_id) -> PlanSnapshot | None`).
Checkpoint write failures are logged as warnings and never block plan
execution.

## Security — Prompt Injection Guard

`browser_*` and `mcp.<server>.<tool>` return content from external, untrusted
sources (web pages, third-party MCP servers). Without defense, an attacker can
embed directives like `IGNORE PREVIOUS INSTRUCTIONS...` in a page or an MCP
response and hijack the next LLM turn.

The engine ships a defense-in-depth guard with two layers:

1. **Content sanitizer** (`engine.sanitizer.DefaultSanitizer`)
   - `ToolSpec.trust_level = "untrusted"` (set on `browser_navigate`,
     `browser_snapshot`, `browser_get_text`, and every bridged MCP tool) flags a
     tool as an untrusted source.
   - Results from untrusted tools are stripped of zero-width characters and
     HTML/XML comments, and common injection patterns
     (`ignore previous instructions`, `SYSTEM:`, `you are now`, `<system>`,
     `[admin mode]`, `bypass safety`, etc.) are replaced by
     `[filtered:prompt-injection-guard]`.
   - The payload is wrapped in
     `<tool_output trust="untrusted">...</tool_output>` fences so the LLM can
     syntactically distinguish data from instructions.

2. **System prompt guard** — when a sanitizer is attached, the engine injects a
   `<security>` section into the system prompt explaining the fence semantics
   and telling the model to treat fenced content as data, never as commands.

### Defaults

- `default_agent()` enables `DefaultSanitizer` unless `sanitizer=None` is passed.
- HTTP server: `PROMPT_INJECTION_GUARD=true` (env) / `prompt_injection_guard: True`
  (config) — the server factory wires a `DefaultSanitizer` into every agent it
  creates.
- Trusted tools (`ToolSpec.trust_level = "trusted"`, the default) always pass
  through unmodified.

### Disabling

```python
agent = default_agent(provider=..., model=..., sanitizer=None)
```

or

```bash
PROMPT_INJECTION_GUARD=false uv run topsport-agent-serve
```

### Caveats

The guard is defense-in-depth, not a silver bullet:

- A sufficiently disguised instruction may bypass the regex set.
- LLMs can still be confused; the `<security>` guidance is advisory.
- Do **not** remove human-in-the-loop for high-privilege tools because a
  sanitizer is enabled. Layer approvals for destructive actions.

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

## Claude Code Plugin Ecosystem

topsport-agent can load the full Claude Code plugin ecosystem from
`~/.claude/plugins/`, supporting all four extension types.

### Supported Extension Types

| Type | Source | Integration |
| --- | --- | --- |
| Skills | `plugins/cache/*/skills/*/SKILL.md` | Registered in `SkillRegistry` as `plugin:skill` |
| Commands | `plugins/cache/*/commands/*.md` | Converted to skills as `plugin:command` |
| Agents | `plugins/cache/*/agents/*.md` | `list_agents` / `spawn_agent` tools |
| Hooks | `plugins/cache/*/hooks/hooks.json` | `EventSubscriber` bridging to shell commands |

### Usage

```python
from topsport_agent.plugins import PluginManager
from topsport_agent.skills import SkillRegistry

mgr = PluginManager()
mgr.load()

# Skills: local dirs first (higher priority), plugin dirs after
local_dirs = [Path.home() / ".claude" / "skills"]
registry = SkillRegistry(local_dirs + mgr.skill_dirs())
registry.load()

# Agents: expose as tools
agent_tools = build_agent_tools(mgr.agent_registry())

# Hooks: inject as event subscriber
engine = Engine(
    provider, tools=[...],
    config=config,
    event_subscribers=[mgr.hook_runner()],
)

# Cleanup temp dirs on exit
mgr.cleanup()
```

### Plugin Discovery

Reads `~/.claude/plugins/installed_plugins.json` to find all installed
plugins. Only explicitly installed plugins are loaded. Naming convention:
`plugin_name:extension_name` (e.g. `superpowers:brainstorming`).

### Priority

Local `~/.claude/skills/` always wins over plugin skills on name collision.
Among plugins, first discovered wins (sorted by marketplace then name).

### Hook Event Mapping

| Claude Hook Event | Engine EventType | Matcher target |
| --- | --- | --- |
| SessionStart | `RUN_START` | always |
| SessionEnd | `RUN_END` | always |
| PreToolUse | `TOOL_CALL_START` | `payload["name"]` |
| PostToolUse | `TOOL_CALL_END` | `payload["name"]` |
| UserPromptSubmit | `MESSAGE_APPENDED` (role=user) | always |

Hooks execute in subprocess with 30s default timeout. Failures are logged
but never interrupt the engine.

## Not yet implemented

- HTTP or streaming surface for frontend integration
- Plan editing after approval
- Dynamic re-planning during execution
- Shared `common/frontmatter.py` -- memory, skills, and future modules currently
  carry their own copy
