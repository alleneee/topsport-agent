# Multimodal Support via OpenAI-Compatible Scheme

- Date: 2026-04-24
- Status: Proposed
- Author: niko (with brainstorming via Claude)

## 1. Context

`topsport-agent` currently models messages as plain text:
`Message.content: str | None`. Both adapters (`AnthropicMessagesAdapter`,
`OpenAIChatAdapter`) consume only text input/output. There is no first-class
support for image or video analysis, nor for image generation.

The target deployment uses an **OpenAI-compatible endpoint** (not the official
OpenAI API directly). Popular candidates include Alibaba DashScope's Qwen-VL
series, one-api/new-api proxy layers, and self-hosted inference servers
(vLLM, SGLang). The unifying contract is the OpenAI Chat Completions request
schema and (where supported) the OpenAI Images Generation schema.

This spec adds three capabilities behind that single scheme constraint:

1. **Image analysis** — user messages may contain image content parts
2. **Video analysis** — user messages may contain video content parts
   (Qwen-VL-specific but OpenAI-scheme compatible)
3. **Image generation** — independent synchronous API, decoupled from the
   Engine's LLM loop

The `AnthropicMessagesAdapter` is **not** modified in this spec.

## 2. Non-Goals

Explicitly **out of scope** to keep this spec focused:

- Anthropic multimodal support (native format differs; separate spec later)
- Audio input analysis (OpenAI `input_audio` content part) — YAGNI
- Assistant-side multimodal output (images in LLM responses) — requires the
  Responses API and is not used by the selected providers
- Asynchronous image generation (polling-based providers like Alibaba wanx
  native endpoint) — users needing this must configure a sync-compatible
  proxy
- Image frame-sequence video input (`{"type": "video", "video": [urls...]}`)
  — Qwen-VL-specific format, not OpenAI scheme
- OpenAI Files API (`file_id`) references
- Image preprocessing (resize, compression, format conversion)
- Image generation as a builtin tool the LLM can invoke — explicitly chosen
  "independent API" form-factor
- `default_agent()` auto-wiring image generation — must be configured
  explicitly

## 3. Architecture Overview

### Provider constraint

Both data-plane features follow the OpenAI scheme:

- Chat Completions → `POST {base_url}/chat/completions` with
  `ChatCompletionContentPart` arrays in `user` messages
- Images Generation → `POST {base_url}/images/generations`, synchronous

Users configure `base_url` + `api_key` on the underlying `AsyncOpenAI` client.
If a specific endpoint does not support a feature (e.g., DashScope does not
currently support `/images/generations` via its compat endpoint), the
response is a provider-level error passed through without translation. No
provider-specific code paths exist in this codebase.

### Data flow — input side

```
User code
   │
   │  Message(role=USER, content_parts=[
   │      TextPart("What is in this image?"),
   │      ImagePart(source=MediaRef(url="https://..."))
   │  ])
   ▼
Engine.run → LLMRequest → OpenAIChatAdapter.build_payload
   │
   │  _convert_messages detects content_parts,
   │  emits array-form content;
   │  local files are read and base64-encoded into data URIs.
   ▼
OpenAI-compatible /v1/chat/completions
   │
   ▼  Response contains text blocks only → LLMResponse.text
```

### Data flow — image generation (independent of Engine)

```
User code
   ▼
agent.generate_image(prompt="a cat", size="1024x1024")
   │
   ▼
OpenAIImageGenerationClient.generate(request)
   │
   ▼  POST /v1/images/generations   ← synchronous await
   │
   ▼
ImageGenerationResponse(images=[GeneratedImage(url=...)])
```

### Invariants preserved

- `Message.content: str | None` remains fully backward compatible. Code not
  using `content_parts` is zero-change.
- "Assistant message with `tool_calls` must be immediately followed by
  `tool_results`" is untouched.
- `AnthropicMessagesAdapter` is not modified; all Anthropic tests remain
  green.
- Optional-dependency convention: `openai` and `httpx` are only imported
  via `importlib.import_module(variable_name)` at runtime; no top-level
  test imports.

## 4. Data Model

All new types live in `src/topsport_agent/types/message.py`.

### `MediaRef` — unified media source

```python
@dataclass(slots=True, frozen=True)
class MediaRef:
    """Image/video source. Exactly one of url/path/data must be non-None.

    media_type is optional; for `path`, the file extension is used. For
    `data`, media_type is mandatory (bytes cannot be reliably inferred).
    """
    url: str | None = None
    path: Path | None = None
    data: bytes | None = None
    media_type: str | None = None   # e.g. "image/png", "video/mp4"

    def __post_init__(self) -> None:
        provided = sum(x is not None for x in (self.url, self.path, self.data))
        if provided != 1:
            raise ValueError("MediaRef requires exactly one of url/path/data")
        if self.data is not None and self.media_type is None:
            raise ValueError("MediaRef(data=...) requires media_type explicitly")
```

### `ContentPart` — discriminated union

```python
@dataclass(slots=True, frozen=True)
class TextPart:
    text: str

@dataclass(slots=True, frozen=True)
class ImagePart:
    source: MediaRef
    detail: Literal["auto", "low", "high"] = "auto"

@dataclass(slots=True, frozen=True)
class VideoPart:
    source: MediaRef

ContentPart = TextPart | ImagePart | VideoPart
```

Discrimination is via `isinstance()` in the adapter layer — idiomatic Python
and type-checker friendly. Frozen for safe cross-context passing.

### `Message` extension

```python
@dataclass(slots=True)
class Message:
    role: Role
    content: str | None = None
    content_parts: list[ContentPart] | None = None   # NEW
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
```

Coexistence rule (enforced in the OpenAI adapter):

| `content_parts` | `content` | Role | Behavior |
|-----------------|-----------|------|----------|
| `None` | any | any | Legacy path: plain-text (or empty) |
| `[...]` | `None` | `USER` | New path: array content |
| `[...]` | `"..."` | `USER` | `content` prepended as a leading `TextPart`; then `content_parts` appended |
| `[...]` | any | `ASSISTANT` / `SYSTEM` / `TOOL` | Adapter raises `ValueError` (OpenAI scheme does not accept array content on these roles) |

### Convenience constructors

```python
def image_url(url: str, *, detail: str = "auto") -> ImagePart: ...
def image_file(path: str | Path, *, detail: str = "auto") -> ImagePart: ...
def image_bytes(data: bytes, media_type: str, *, detail: str = "auto") -> ImagePart: ...
def video_url(url: str) -> VideoPart: ...
def video_file(path: str | Path) -> VideoPart: ...
```

## 5. OpenAI Adapter Extension

File: `src/topsport_agent/llm/adapters/openai_chat.py`

Only the `Role.USER` branch in `_convert_messages` changes. `SYSTEM`,
`ASSISTANT`, `TOOL` branches are unchanged.

### USER branch logic

```python
if msg.role == Role.USER:
    if msg.content_parts is None:
        converted.append({"role": "user", "content": msg.content or ""})
        continue

    parts_payload: list[dict[str, Any]] = []
    if msg.content:
        parts_payload.append({"type": "text", "text": msg.content})
    for part in msg.content_parts:
        parts_payload.append(self._encode_content_part(part))
    converted.append({"role": "user", "content": parts_payload})
    continue
```

### Non-USER roles with content_parts → error

```python
if msg.role in (Role.ASSISTANT, Role.SYSTEM, Role.TOOL) and msg.content_parts is not None:
    raise ValueError(
        f"{msg.role.value} messages do not support content_parts (OpenAI scheme)"
    )
```

Rationale: OpenAI Chat Completions only accepts array-form content on `user`
messages. Silent acceptance would discard content and produce confusing LLM
behavior.

### `_encode_content_part`

```python
def _encode_content_part(self, part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        url_value = self._resolve_media_url(part.source, default_media="image/png")
        return {
            "type": "image_url",
            "image_url": {"url": url_value, "detail": part.detail},
        }
    if isinstance(part, VideoPart):
        url_value = self._resolve_media_url(part.source, default_media="video/mp4")
        return {"type": "video_url", "video_url": {"url": url_value}}
    raise TypeError(f"Unsupported content part: {type(part).__name__}")
```

### `_resolve_media_url`

```python
_EXT_TO_MEDIA: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}

@staticmethod
def _resolve_media_url(ref: MediaRef, *, default_media: str) -> str:
    if ref.url is not None:
        return ref.url
    if ref.path is not None:
        suffix = ref.path.suffix.lower()
        media_type = ref.media_type or _EXT_TO_MEDIA.get(suffix)
        if not media_type:
            raise ValueError(
                f"Cannot infer media_type from {ref.path!r}; "
                "pass MediaRef(media_type=...)"
            )
        raw = ref.path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{media_type};base64,{b64}"
    if ref.data is not None:
        assert ref.media_type is not None  # guaranteed by __post_init__
        b64 = base64.b64encode(ref.data).decode("ascii")
        return f"data:{ref.media_type};base64,{b64}"
    raise ValueError("MediaRef has no url/path/data")  # unreachable
```

### Payload example

User input:

```python
Message(role=Role.USER, content_parts=[
    TextPart("Compare these two images"),
    image_url("https://cdn.example.com/a.jpg"),
    image_file("/tmp/b.png", detail="high"),
])
```

Adapter output (single OpenAI message):

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Compare these two images"},
    {"type": "image_url",
     "image_url": {"url": "https://cdn.example.com/a.jpg", "detail": "auto"}},
    {"type": "image_url",
     "image_url": {"url": "data:image/png;base64,iVBORw0KGgo...",
                   "detail": "high"}}
  ]
}
```

### Error matrix

| Scenario | Behavior |
|----------|----------|
| `MediaRef` with zero or multiple source fields | `MediaRef.__post_init__` raises `ValueError` |
| `MediaRef(data=b"...")` without `media_type` | `__post_init__` raises |
| Local path with unknown extension and no `media_type` | adapter raises `ValueError` |
| Local path does not exist | native `FileNotFoundError` from `Path.read_bytes` |
| `ASSISTANT`/`SYSTEM`/`TOOL` role with `content_parts` | adapter raises `ValueError` |
| Unknown `ContentPart` subtype (future extension gap) | adapter raises `TypeError` |

## 6. Image Generation Subsystem

New file: `src/topsport_agent/llm/image_generation.py`. Single-file start;
split into a subpackage only if complexity warrants.

### Data model

```python
@dataclass(slots=True)
class ImageGenerationRequest:
    prompt: str
    model: str
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    n: int = 1
    response_format: Literal["url", "b64_json"] = "url"
    provider_options: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True, frozen=True)
class GeneratedImage:
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None

    async def save(
        self,
        path: str | Path,
        *,
        http_client: Any | None = None,
    ) -> Path: ...

@dataclass(slots=True)
class ImageGenerationResponse:
    images: list[GeneratedImage]
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
```

### `OpenAIImageGenerationClient`

```python
class OpenAIImageGenerationClient:
    """Synchronous-request client for OpenAI /v1/images/generations."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        default_model: str | None = None,
    ) -> None:
        if client is None and client_factory is None:
            raise ValueError("Either `client` or `client_factory` is required")
        self._client = client
        self._client_factory = client_factory
        self.default_model = default_model

    async def generate(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        client = self._ensure_client()
        kwargs = self._build_kwargs(request)
        raw = await client.images.generate(**kwargs)
        return self._parse(raw)

    def _ensure_client(self) -> Any:
        if self._client is None:
            assert self._client_factory is not None
            self._client = self._client_factory()
        return self._client

    @staticmethod
    def _build_kwargs(req: ImageGenerationRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": req.model,
            "prompt": req.prompt,
            "n": req.n,
            "response_format": req.response_format,
        }
        if req.size: kwargs["size"] = req.size
        if req.quality: kwargs["quality"] = req.quality
        if req.style: kwargs["style"] = req.style
        kwargs.update(req.provider_options)   # overrides take effect last
        return kwargs

    @staticmethod
    def _parse(raw: Any) -> ImageGenerationResponse:
        data = getattr(raw, "data", None) or []
        images = [
            GeneratedImage(
                url=getattr(item, "url", None),
                b64_json=getattr(item, "b64_json", None),
                revised_prompt=getattr(item, "revised_prompt", None),
            )
            for item in data
        ]
        usage: dict[str, int] = {}
        usage_obj = getattr(raw, "usage", None)
        if usage_obj is not None:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                val = getattr(usage_obj, key, None)
                if val is not None:
                    usage[key] = int(val)
        return ImageGenerationResponse(images=images, usage=usage, raw=raw)
```

### `GeneratedImage.save`

```python
async def save(
    self,
    path: str | Path,
    *,
    http_client: Any | None = None,
) -> Path:
    target = Path(path)
    if self.b64_json is not None:
        target.write_bytes(base64.b64decode(self.b64_json))
        return target
    if self.url is not None:
        mod_name = "httpx"
        httpx_mod = importlib.import_module(mod_name)   # lazy import
        if http_client is None:
            async with httpx_mod.AsyncClient() as c:
                resp = await c.get(self.url)
                resp.raise_for_status()
                target.write_bytes(resp.content)
        else:
            resp = await http_client.get(self.url)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        return target
    raise ValueError("GeneratedImage has neither url nor b64_json")
```

## 7. Agent Integration

File: `src/topsport_agent/agent/base.py`

### `Agent.run()` accepts multimodal input

```python
async def run(
    self,
    user_input: str | list[ContentPart] | Message,
    session: Session,
) -> AsyncIterator[Event]:
    if isinstance(user_input, Message):
        msg = user_input
    elif isinstance(user_input, list):
        msg = Message(role=Role.USER, content_parts=user_input)
    else:
        msg = Message(role=Role.USER, content=user_input)
    session.messages.append(msg)
    session.state = RunState.IDLE
    async for event in self._engine.run(session):
        yield event
    self._engine.reset_cancel()
```

### `AgentConfig` and `Agent.__init__` accept `image_generator`

```python
@dataclass
class AgentConfig:
    ...existing fields...
    image_generator: "OpenAIImageGenerationClient | None" = None

class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        config: AgentConfig,
        *,
        engine: Engine,
        ...existing params...,
        image_generator: "OpenAIImageGenerationClient | None" = None,
    ) -> None:
        ...
        self._image_generator = image_generator
        self._capability_bundle["image_generator"] = image_generator
```

### `Agent.generate_image()` convenience method

```python
async def generate_image(
    self,
    prompt: str,
    *,
    model: str | None = None,
    **kwargs: Any,
) -> "ImageGenerationResponse":
    if self._image_generator is None:
        raise RuntimeError(
            "No image_generator configured; pass image_generator= "
            "to Agent(), or construct OpenAIImageGenerationClient directly."
        )
    resolved_model = model or self._image_generator.default_model
    if resolved_model is None:
        raise ValueError(
            "model required (or set default_model on image_generator)"
        )
    request = ImageGenerationRequest(
        prompt=prompt, model=resolved_model, **kwargs
    )
    return await self._image_generator.generate(request)
```

### `Agent.from_config` forwards image_generator

```python
return cls(
    provider=provider,
    config=config,
    engine=engine,
    cleanup_callbacks=cleanup_callbacks,
    skill_registry=skill_registry,
    plugin_manager=plugin_manager,
    capability_bundle=capability_bundle,
    image_generator=config.image_generator,
)
```

### `spawn_child` note (no inheritance required)

`Agent.spawn_child` returns a `(Session, Engine)` tuple, not a child `Agent`.
Since `generate_image` is an `Agent`-level convenience method, there is no
child object on which to call it — image generation is not part of the
Engine/tool execution surface. Callers who need image generation inside a
sub-agent's tool execution must pass the `OpenAIImageGenerationClient`
directly into that tool's closure.

No `_capability_bundle` entry is added for `image_generator`; tracking it
there would be dead state with no consumer.

## 8. Testing Strategy

All new tests MUST pass without `openai` or `httpx` installed. No top-level
`import openai` or `import httpx` in test modules.

### New/extended test files

| File | Status | Cases |
|------|--------|-------|
| `tests/test_message_multimodal.py` | NEW | ~8 |
| `tests/test_openai_adapter.py` | EXTEND | +9 |
| `tests/test_image_generation.py` | NEW | ~10 |
| `tests/test_agent_multimodal.py` | NEW | ~6 |

### Coverage matrix

| Invariant | Test case |
|-----------|-----------|
| `MediaRef` mutually exclusive source fields | `test_media_ref_requires_exactly_one_source` |
| `MediaRef(data=...)` requires media_type | `test_media_ref_bytes_requires_media_type` |
| URL passes through unchanged | `test_user_message_with_image_url` |
| Local file auto-base64 with extension inference | `test_user_message_with_image_file_auto_base64` |
| Raw bytes with explicit media_type | `test_user_message_with_image_bytes_explicit_media_type` |
| `video_url` content part emitted | `test_user_message_with_video_url` |
| Non-USER roles reject content_parts | `test_assistant_role_with_content_parts_raises` |
| **Backward compat: plain `content: str` unchanged** | `test_user_message_plain_string_payload_unchanged` |
| Image generation sync call builds kwargs and parses response | `test_generate_builds_kwargs_and_parses_response` |
| `GeneratedImage.save` URL path with injected http_client | `test_save_url_uses_injected_http_client` |
| `GeneratedImage.save` b64_json decoding | `test_save_b64_writes_decoded_bytes` |

### Verification

- `uv run pytest -v` — full suite green
- `uv pip uninstall openai httpx -y && uv run pytest -v` — new tests green
  (verifies optional-dep independence)
- grep test modules for top-level `import openai` / `import httpx` — none

## 9. File Layout and Landing Order

```
src/topsport_agent/
├── types/
│   └── message.py                    [MODIFY]
├── llm/
│   ├── adapters/
│   │   └── openai_chat.py            [MODIFY]
│   └── image_generation.py           [NEW]
└── agent/
    └── base.py                       [MODIFY]

tests/
├── test_message_multimodal.py        [NEW]
├── test_openai_adapter.py            [EXTEND]
├── test_image_generation.py          [NEW]
└── test_agent_multimodal.py          [NEW]
```

Files explicitly not touched:

- `llm/adapters/anthropic.py`
- `llm/clients/*`, `llm/providers/*`
- `engine/*`
- `agent/default.py`
- `types/message.py::ToolCall`, `::ToolResult`, `::Role`

### Landing order (TDD, dependency topology)

1. **Task 1 — Type layer** (`types/message.py`). No dependencies.
2. **Task 2 — Adapter extension** (`llm/adapters/openai_chat.py`). Depends on Task 1.
3. **Task 3 — Image generation module** (`llm/image_generation.py`). No dependencies.
4. **Task 4 — Agent integration** (`agent/base.py`). Depends on Tasks 1 and 3.
5. **Task 5 — Documentation.** README section on multimodal usage; new
   learnings entries in `.learnings/LEARNINGS.md` as they emerge.

Tasks 1 and 3 are parallelizable. Tasks 2 and 4 serialize behind their
dependencies.

### Definition of Done

- [ ] `uv run pytest -v` fully green
- [ ] No top-level `import openai` / `import httpx` in any test module
- [ ] README section on multimodal input and image generation usage,
      markdownlint-clean
- [ ] Existing `tests/test_openai_adapter.py` cases unchanged
      (backward-compat proof)
- [ ] `tests/test_anthropic_adapter.py` unchanged
- [ ] Pyright reports no new errors

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OpenAI-compatible endpoints differ on `video_url` support | Medium | Document that Qwen-VL or equivalent is required; provider 4xx errors pass through unchanged |
| base64 encoding inflates memory for large video files | Medium | Document URL form as preferred for >1MB media; no compression in MVP |
| `data:` URI exceeds provider payload limits | Low | Provider error is passed through; caller decides to switch to URL |
| OpenAI SDK version differences in `response_format` semantics | Low | `provider_options` allows full override; test coverage on default behavior |
| User configures DashScope compat endpoint and calls `generate_image`, gets 404 | Medium | Document known limitations; surface provider error with original status |

## 11. Glossary

- **OpenAI-compatible endpoint**: An HTTP endpoint that implements the
  OpenAI REST API schema at a user-controlled `base_url`. Examples:
  `https://api.openai.com/v1`, DashScope's compat layer
  `https://dashscope.aliyuncs.com/compatible-mode/v1`, self-hosted proxies
  like one-api/new-api, inference servers like vLLM.
- **Content part**: A single item in an OpenAI `user` message's `content`
  array. Types: `text`, `image_url`, `input_audio`, `video_url` (Qwen-VL
  extension), `file`.
- **data URI**: Inline base64-encoded payload with a media type, of the form
  `data:<media_type>;base64,<b64>`. Both OpenAI and DashScope accept this in
  the `image_url.url` field.
- **spawn_child**: The mechanism by which an Agent creates sub-agents that
  inherit its configured capabilities.
