# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and test

```bash
uv sync                          # install dev dependencies
uv sync --group llm              # + anthropic, openai
uv sync --group tracing          # + langfuse
uv sync --group mcp              # + mcp, httpx
uv run pytest -v                 # run all tests
uv run pytest -k 'test_name'     # run a single test
```

## Testing rules

- All tests MUST pass without optional dependency groups installed. Use mock clients and injectable `session_factory` / `client=` parameters. Never `import anthropic`, `import openai`, `import langfuse`, or `import mcp` at test module level.
- Run the full test suite after any code change. Do not report work as complete until `uv run pytest -v` shows all green.

## Architecture invariants

These are tested and must not be violated:

- An assistant message with `tool_calls` must be immediately followed by `tool_result` messages. No other message types in between. Violating this causes 400 errors from both Anthropic and OpenAI.
- `ContextProvider.provide()` output is ephemeral: merged into the LLM call but never appended to `session.messages`. Re-injection every step is intentional.
- Builtin tool names (passed via `Engine(tools=...)`) always win over `ToolSource` dynamic names on collision.
- `Engine.cancel()` preempts in-flight LLM calls via `asyncio.wait(FIRST_COMPLETED)`, not polling.
- Tool list is snapshotted per step via `_snapshot_tools()`. Do not cache across steps.
- Event subscribers receive events in yield order. Exceptions in one subscriber must not affect the engine or other subscribers.

## Development workflow

- Phase discipline: outline, first draft, polish. Show results after each phase before continuing.
- Update README.md (markdownlint compliant) when a module is completed.
- Record non-obvious discoveries in `.learnings/LEARNINGS.md` using the Context / Learned / Evidence format. See @.learnings/LEARNINGS.md for existing entries.

## Non-obvious patterns

- Optional dependencies use `importlib.import_module(variable_name)` (not string literal) to bypass Pyright `reportMissingImports`. Example: `mod_name = "anthropic"; mod = importlib.import_module(mod_name)`.
- Anthropic adapter merges consecutive `Role.TOOL` messages into a single `user` message with `tool_result` content blocks. OpenAI adapter keeps each as an independent `role=tool` message.
- OpenAI tool call `arguments` is a JSON string; adapter does `json.dumps` outbound and `json.loads` inbound with a `{"_raw_arguments": ...}` fallback for malformed JSON.
- System messages: Anthropic lifts them to the top-level `system` parameter; OpenAI keeps them as `role=system` messages in the array. Engine's `_build_call_messages` collapses all ephemeral system messages with `session.system_prompt` before handing to the adapter.
