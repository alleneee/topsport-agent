# Examples

Runnable code samples covering the core features of `topsport-agent`.

All examples use `AnthropicProvider`. To switch to OpenAI, replace the import
with `from topsport_agent.llm.providers.openai_chat import OpenAIChatProvider`
and set `OPENAI_API_KEY` in the environment.

## Prerequisites

```bash
uv sync --group llm
# only example 04 needs:
uv sync --group llm --group mcp
```

Export your API key before running any sample:

```bash
export ANTHROPIC_API_KEY=sk-...
```

## Index

| File | Demonstrates |
|------|--------------|
| `smoke_test.py` | Bare-bones `Engine` + `ToolSpec` sanity check |
| `01_minimal_tool.py` | Minimum viable `Engine` with a custom `add` tool |
| `02_default_agent.py` | High-level `default_agent()` with skills / memory / plugins / file ops |
| `03_event_subscriber.py` | Custom `EventSubscriber` that measures LLM / tool latency |
| `04_mcp_tool_source.py` | Dynamic tool source via `MCPManager` reading a Claude Desktop config |
| `05_cancel.py` | Cooperative cancellation through `Engine.cancel()` / `FIRST_COMPLETED` |
| `06_streaming.py` | Token streaming via `LLM_TEXT_DELTA` events (`stream=True`) |
| `07_context_provider.py` | Ephemeral per-step context injection without polluting `session.messages` |

## Core extension points (quick reference)

| Entry point | Where to inject | Invoked |
|-------------|-----------------|---------|
| `tools=[ToolSpec]` | `Engine.__init__` / `AgentConfig.extra_tools` | merged per step |
| `tool_sources=[ToolSource]` | `Engine.__init__` / `AgentConfig.extra_tool_sources` | `list_tools()` each step |
| `context_providers=[ContextProvider]` | `Engine.__init__` / `AgentConfig.extra_context_providers` | before each LLM call (ephemeral) |
| `post_step_hooks=[PostStepHook]` | `Engine.__init__` / `AgentConfig.extra_post_step_hooks` | after each step |
| `event_subscribers=[EventSubscriber]` | `Engine.__init__` / `AgentConfig.extra_event_subscribers` | every event, in order |

Invariants worth remembering (from `CLAUDE.md`):

- Assistant messages with `tool_calls` must be followed immediately by `tool_result` messages.
- `ContextProvider.provide()` output is **ephemeral**: merged into the LLM call, never persisted to `session.messages`.
- Builtin tool names always win over `ToolSource` dynamic names on collision.
- Tool list is snapshotted per step — never cached across steps.
- Event subscriber exceptions do not affect the engine or other subscribers.
