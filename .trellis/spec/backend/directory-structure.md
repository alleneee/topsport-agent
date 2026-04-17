# Directory Structure

> How the agent runtime code is organized.

---

## Overview

Single-package Python project with a `src/` layout managed by `hatchling`.
All source code lives under `src/topsport_agent/`. Tests live in `tests/`.

---

## Directory Layout

```
src/topsport_agent/
├── types/              Message, ToolCall, ToolResult, Session, RunState, Event
├── llm/
│   ├── provider.py     LLMProvider + StreamingLLMProvider Protocols
│   ├── request.py      LLMRequest dataclass
│   ├── response.py     LLMResponse + ProviderResponseMetadata
│   ├── stream.py       LLMStreamChunk (incremental delta type)
│   ├── clients/        SDK client wrappers (anthropic_messages, openai_chat)
│   ├── providers/      Provider orchestration (adapter + client composition)
│   └── adapters/       Payload/response codecs per provider
├── engine/
│   ├── loop.py         Engine + ReAct main loop + cancel + event dispatch + streaming
│   ├── hooks.py        ContextProvider / ToolSource / PostStepHook / EventSubscriber
│   ├── prompt.py       PromptBuilder + SectionPriority (XML-tagged sections)
│   ├── compaction/     micro (clear old tool results) + auto (LLM summary) + hook
│   ├── planner.py      Multi-agent Planner (creates DAG of PlanSteps)
│   ├── orchestrator.py Plan executor: topological waves, sub-agent isolation
│   ├── loop_detector.py    Repeated tool call detection + wrap
│   ├── interject_queue.py  Queue messages during tool execution
│   └── concurrency.py      EngineGuard + guarded_run wrapper
├── memory/             Session-scoped working memory (FileMemoryStore, injector, tools)
├── skills/             Anthropic Agent Skills spec (registry, loader, matcher, injector, tools)
├── browser/            Playwright-based browser control with @ref snapshot interaction
├── mcp/                MCP client (JSON config, lazy session, tool bridge, meta tools)
├── plugins/            Claude Code plugin ecosystem
│   ├── discovery.py    Scan ~/.claude/plugins/installed_plugins.json
│   ├── plugin.py       PluginDescriptor scanning (skills/commands/agents/hooks paths)
│   ├── agent_registry.py  AgentDefinition + AgentRegistry + build_agent_tools + SpawnExecutor
│   ├── hook_runner.py  EventSubscriber bridging Engine events to plugin hooks.json
│   └── manager.py      PluginManager: unified entry point
├── agent/              High-level Agent abstraction
│   ├── base.py         Agent + AgentConfig + from_config + _build_spawn_executor
│   ├── default.py      default_agent() preset (all capabilities on)
│   └── browser.py      browser_agent() preset (browser mandatory + focused prompt)
├── observability/      Tracer alias, NoOpTracer, LangfuseTracer
├── tools/              ToolExecutor, safe_shell, FileBlobStore, file_ops (read/write/edit/grep/glob)
└── cli/                Interactive REPL, builtin tools (echo/calc/current_time)
```

---

## Module Organization

Each top-level module is a self-contained package with its own `__init__.py`.
Dependencies flow downward:

```
agent   →  engine, llm, memory, skills, plugins, tools, browser (optional)
engine  →  llm, types
memory  →  types
skills  →  types
browser →  types (lazy playwright import)
mcp     →  types, llm (for meta_tools ToolSpec)
plugins →  types, skills (for _frontmatter parser reuse)
tools   →  types
observability  →  types, engine.hooks
cli     →  agent, types
```

No circular imports. `engine` never imports from `memory`, `skills`, `mcp`,
`plugins`, or `observability` directly. Integration happens through
Protocol-based hooks. The `agent/` package is the single assembly point that
wires these modules together via `Agent.from_config`.

---

## Naming Conventions

- Modules: `snake_case.py`
- Internal helpers: prefix with `_` (e.g. `_frontmatter.py`)
- Classes: `PascalCase`
- Protocols: named by role (`LLMProvider`, `ContextProvider`, `BlobStore`)
- Factory functions: `build_*_tools(...)` returns `list[ToolSpec]`
- Adapters for optional deps: lazy `importlib.import_module(variable)` pattern

---

## Examples

- Well-structured module: `memory/` (types, store protocol, file impl, injector, tools)
- Protocol-based hook: `engine/hooks.py` (4 protocols, all used by Engine.__init__)
- Adapter layer: `llm/adapters/anthropic.py` (payload build + response parse, no SDK import)
