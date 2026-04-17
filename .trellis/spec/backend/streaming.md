# LLM Streaming Contract

> How streaming LLM responses flow from provider to CLI.

---

## Overview

The framework supports optional streaming output. Non-streaming path
(`LLMProvider.complete`) remains the default. Streaming is opt-in via
`EngineConfig.stream=True` AND provider-side `StreamingLLMProvider` Protocol.

---

## Protocol Contract

### StreamingLLMProvider (`src/topsport_agent/llm/provider.py`)

```python
@runtime_checkable
class StreamingLLMProvider(Protocol):
    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]: ...
```

- Implementations MUST use `async def stream(...)` + `yield` (async generator).
- Detection is via `isinstance(provider, StreamingLLMProvider)` at runtime.
- `complete()` remains mandatory on all providers; `stream()` is an extension.

### LLMStreamChunk (`src/topsport_agent/llm/stream.py`)

```python
@dataclass(slots=True)
class LLMStreamChunk:
    type: Literal["text_delta", "tool_call_delta", "done"]
    text_delta: str | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_delta_args: str | None = None
    final_response: LLMResponse | None = None
```

**Invariants:**
- `text_delta` chunks carry incremental text (NOT cumulative).
- Exactly one `done` chunk MUST be emitted at end, with `final_response` set.
- If `done` is missing, Engine synthesizes an empty error response to avoid deadlock.
- Tool-call deltas are optional; current providers only emit complete tool calls in `final_response`.

---

## Engine Integration

`Engine._stream_llm_events()` (`src/topsport_agent/engine/loop.py`):

```python
async def _stream_llm_events(
    self,
    messages: list[Message],
    tools: list[ToolSpec],
    session: Session,
    step: int,
    final_holder: list[LLMResponse],
) -> AsyncIterator[Event]:
```

- Mutable `final_holder: list[LLMResponse]` is the return channel for the
  aggregated response. Caller reads `final_holder[0]` after generator exhausts.
- Yields `Event(type=LLM_TEXT_DELTA, payload={"step", "delta"})` for each text chunk.
- Checks `self._cancel_event` before each chunk; raises `Cancelled()` on set.
- Calls `stream.aclose()` in `finally` to release provider connection.

**Why list-based return:** Python async generators cannot `return` values.
Alternatives (asyncio.Queue, instance state, return-via-exception) are more
complex or stateful. `list[T]` is idiomatic and thread-safe within one task.

---

## Event Contract

### EventType.LLM_TEXT_DELTA

```python
Event(
    type=EventType.LLM_TEXT_DELTA,
    session_id=session.id,
    payload={"step": int, "delta": str},
)
```

- Emitted ONLY in streaming mode.
- `delta` is always non-empty (empty deltas are filtered in `_stream_llm_events`).
- Order preserved: deltas arrive in the same order the model generates them.

### EventType.LLM_CALL_START payload extension

Adds `"stream": bool` flag indicating whether streaming path was taken.
Subscribers can use this to disambiguate trace spans.

---

## Provider Implementations

### Anthropic (`src/topsport_agent/llm/clients/anthropic_messages.py`)

Uses SDK's `client.messages.stream(...)` async context manager.
- Client yields abstract events: `{"type": "text_delta", "text": ...}`
  and `{"type": "final_message", "message": <sdk response>}`.
- Provider translates these to `LLMStreamChunk` and runs `adapter.parse_response()`
  on the final message, so streaming and non-streaming paths produce identical
  `LLMResponse` structures.
- Streaming does NOT retry on failure (connection resume complexity is rejected).

### OpenAI (`src/topsport_agent/llm/clients/openai_chat.py`)

Uses `chat.completions.create(stream=True, stream_options={"include_usage": True})`.
- No built-in `get_final_message()`; client manually aggregates text_parts,
  tool_calls (keyed by `index`), usage, and model name.
- Aggregated result is wrapped in `SimpleNamespace` matching the non-streaming
  response shape, so `adapter.parse_response()` is reused verbatim.
- Tool call `arguments` is concatenated as a string across deltas.

---

## CLI Integration

`cli/main.py::_run_loop`:

1. `default_agent(..., stream=True)` enables streaming.
2. In the event loop, `LLM_TEXT_DELTA` events are printed via
   `console.print(delta, end="", markup=False, highlight=False)`.
3. `streaming_started` flag prevents double-rendering the final Markdown block.
4. `--no-stream` CLI flag opts out (falls back to batched Markdown render).

---

## Validation Matrix

| Case | stream config | Provider has `stream()` | Engine path | LLM_TEXT_DELTA emitted |
| --- | --- | --- | --- | --- |
| Default | False | — | `complete()` | No |
| Opt-in + supported | True | Yes | `stream()` via `_stream_llm_events` | Yes |
| Opt-in + unsupported | True | No | `complete()` (fallback) | No |
| Opt-in + cancelled mid-stream | True | Yes | `Cancelled` raised, stream closed | Partial |

---

## Forbidden Patterns

- DO NOT yield empty `text_delta` (filter at the source).
- DO NOT send `final_response=None` in a `done` chunk.
- DO NOT call `provider.complete()` inside a `stream()` implementation (causes
  double billing).
- DO NOT mutate session.messages from within `_stream_llm_events` — leave that
  to the caller in `_run_inner` after `final_holder[0]` is populated.

---

## Tests

Reference tests in `tests/test_streaming.py`:
- `test_stream_chunks_are_deltas_not_cumulative`: verifies delta semantics
- `test_engine_emits_text_delta_events_when_streaming`: event ordering
- `test_engine_falls_back_to_complete_when_stream_disabled`: opt-in behavior
- `test_engine_stream_skipped_when_provider_does_not_support`: fallback
- `test_engine_stream_handles_tool_calls_in_final_response`: tool-use path
