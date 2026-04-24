# Multimodal via OpenAI-Compatible Scheme — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add image + video analysis (user-side `content_parts` in `Message`) and synchronous image generation (`OpenAIImageGenerationClient` + `Agent.generate_image`) behind the OpenAI-compatible API scheme.

**Architecture:** New types (`MediaRef`, `TextPart`, `ImagePart`, `VideoPart`) in `types/message.py`; `Message.content_parts: list[ContentPart] | None` extends the message without breaking backward compat. Only the OpenAI adapter's `USER` branch changes. New `llm/image_generation.py` wraps `/v1/images/generations`. Agent gains a union-typed `run()` input and optional `generate_image()` convenience. Anthropic adapter untouched.

**Tech Stack:** Python 3.11+, dataclasses (no pydantic for content types), pytest / pytest-asyncio, `openai` SDK (optional dep), `httpx` (optional dep, used only by `GeneratedImage.save`).

**Related spec:** `docs/superpowers/specs/2026-04-24-multimodal-openai-compat-design.md`.

**Conventions:**

- All tests run without `openai` or `httpx` installed; mocks only.
- Optional deps are imported via `importlib.import_module(variable_name)` (see `.learnings/LEARNINGS.md`).
- Commit after each task passes.
- Run `uv run pytest -v` end-to-end after the final task to confirm zero regressions.

---

## Task 1: MediaRef dataclass

**Files:**
- Modify: `src/topsport_agent/types/message.py`
- Test: `tests/test_message_multimodal.py` (new)

- [ ] **Step 1: Create failing test file**

```python
# tests/test_message_multimodal.py
from pathlib import Path

import pytest

from topsport_agent.types.message import MediaRef


def test_media_ref_with_url_only_is_valid() -> None:
    ref = MediaRef(url="https://example.com/a.jpg")
    assert ref.url == "https://example.com/a.jpg"
    assert ref.path is None
    assert ref.data is None


def test_media_ref_with_path_only_is_valid() -> None:
    ref = MediaRef(path=Path("/tmp/x.png"))
    assert ref.path == Path("/tmp/x.png")


def test_media_ref_with_bytes_requires_media_type() -> None:
    MediaRef(data=b"raw", media_type="image/png")
    with pytest.raises(ValueError, match="media_type"):
        MediaRef(data=b"raw")


def test_media_ref_rejects_multiple_sources() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MediaRef(url="http://x", path=Path("/y"))


def test_media_ref_rejects_no_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MediaRef()


def test_media_ref_is_frozen() -> None:
    ref = MediaRef(url="https://x")
    with pytest.raises(Exception):  # FrozenInstanceError / AttributeError
        ref.url = "https://y"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `uv run pytest tests/test_message_multimodal.py -v`
Expected: FAIL with `ImportError: cannot import name 'MediaRef' from 'topsport_agent.types.message'`

- [ ] **Step 3: Add `MediaRef` to `types/message.py`**

Add at the top of `src/topsport_agent/types/message.py` after existing imports:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
```

Then insert right before `class Role`:

```python
@dataclass(slots=True, frozen=True)
class MediaRef:
    """Image/video source. Exactly one of url/path/data must be non-None.

    media_type is optional for url/path (path uses extension lookup).
    For data, media_type is mandatory — bytes cannot be reliably inferred.
    """
    url: str | None = None
    path: Path | None = None
    data: bytes | None = None
    media_type: str | None = None

    def __post_init__(self) -> None:
        provided = sum(
            x is not None for x in (self.url, self.path, self.data)
        )
        if provided != 1:
            raise ValueError(
                "MediaRef requires exactly one of url/path/data"
            )
        if self.data is not None and self.media_type is None:
            raise ValueError(
                "MediaRef(data=...) requires media_type explicitly"
            )
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_message_multimodal.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/types/message.py tests/test_message_multimodal.py
git commit -m "feat(types): add MediaRef for multimodal content sources"
```

---

## Task 2: ContentPart discriminated union + convenience constructors

**Files:**
- Modify: `src/topsport_agent/types/message.py`
- Modify: `tests/test_message_multimodal.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_message_multimodal.py`:

```python
from topsport_agent.types.message import (
    ImagePart,
    TextPart,
    VideoPart,
    image_bytes,
    image_file,
    image_url,
    video_file,
    video_url,
)


def test_text_part_holds_text() -> None:
    assert TextPart(text="hi").text == "hi"


def test_image_url_builds_image_part_with_url_source() -> None:
    part = image_url("https://example.com/x.jpg", detail="high")
    assert isinstance(part, ImagePart)
    assert part.source.url == "https://example.com/x.jpg"
    assert part.detail == "high"


def test_image_file_wraps_path() -> None:
    part = image_file("/tmp/x.png")
    assert part.source.path == Path("/tmp/x.png")
    assert part.detail == "auto"


def test_image_bytes_requires_media_type() -> None:
    part = image_bytes(b"raw", "image/png")
    assert part.source.data == b"raw"
    assert part.source.media_type == "image/png"


def test_video_url_builds_video_part() -> None:
    part = video_url("https://example.com/v.mp4")
    assert isinstance(part, VideoPart)
    assert part.source.url == "https://example.com/v.mp4"


def test_video_file_wraps_path() -> None:
    part = video_file("/tmp/v.mp4")
    assert part.source.path == Path("/tmp/v.mp4")
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `uv run pytest tests/test_message_multimodal.py -v`
Expected: FAIL with ImportError for `ImagePart`, etc.

- [ ] **Step 3: Add ContentPart classes + helpers to `types/message.py`**

Insert after the `MediaRef` class (before `class Role`):

```python
@dataclass(slots=True, frozen=True)
class TextPart:
    """Plain text segment in a multimodal user message."""
    text: str


@dataclass(slots=True, frozen=True)
class ImagePart:
    """Image segment. `detail` maps to OpenAI image_url.detail."""
    source: MediaRef
    detail: Literal["auto", "low", "high"] = "auto"


@dataclass(slots=True, frozen=True)
class VideoPart:
    """Video segment. OpenAI scheme emits `video_url` (Qwen-VL extension)."""
    source: MediaRef


ContentPart = TextPart | ImagePart | VideoPart


def image_url(url: str, *, detail: Literal["auto", "low", "high"] = "auto") -> ImagePart:
    """Build an ImagePart from an HTTPS URL."""
    return ImagePart(source=MediaRef(url=url), detail=detail)


def image_file(
    path: str | Path,
    *,
    detail: Literal["auto", "low", "high"] = "auto",
) -> ImagePart:
    """Build an ImagePart from a local file path."""
    return ImagePart(source=MediaRef(path=Path(path)), detail=detail)


def image_bytes(
    data: bytes,
    media_type: str,
    *,
    detail: Literal["auto", "low", "high"] = "auto",
) -> ImagePart:
    """Build an ImagePart from raw bytes + explicit media_type."""
    return ImagePart(
        source=MediaRef(data=data, media_type=media_type),
        detail=detail,
    )


def video_url(url: str) -> VideoPart:
    """Build a VideoPart from an HTTPS URL."""
    return VideoPart(source=MediaRef(url=url))


def video_file(path: str | Path) -> VideoPart:
    """Build a VideoPart from a local file path."""
    return VideoPart(source=MediaRef(path=Path(path)))
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `uv run pytest tests/test_message_multimodal.py -v`
Expected: all tests pass (12 total now).

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/types/message.py tests/test_message_multimodal.py
git commit -m "feat(types): add ContentPart union + convenience constructors"
```

---

## Task 3: Extend Message with content_parts field

**Files:**
- Modify: `src/topsport_agent/types/message.py`
- Modify: `tests/test_message_multimodal.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_message_multimodal.py`:

```python
from topsport_agent.types.message import Message, Role


def test_message_defaults_content_parts_to_none() -> None:
    msg = Message(role=Role.USER, content="hello")
    assert msg.content_parts is None
    assert msg.content == "hello"


def test_message_accepts_content_parts() -> None:
    parts = [TextPart("hi"), image_url("https://x/a.jpg")]
    msg = Message(role=Role.USER, content=None, content_parts=parts)
    assert msg.content_parts == parts
    assert msg.content is None


def test_message_allows_both_content_and_content_parts_dataclass_level() -> None:
    msg = Message(
        role=Role.USER,
        content="lead text",
        content_parts=[image_url("https://x")],
    )
    assert msg.content == "lead text"
    assert msg.content_parts is not None
    assert len(msg.content_parts) == 1
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_message_multimodal.py::test_message_accepts_content_parts -v`
Expected: FAIL — `TypeError: got an unexpected keyword argument 'content_parts'`.

- [ ] **Step 3: Add the field to `Message`**

In `src/topsport_agent/types/message.py`, modify the `Message` dataclass:

```python
@dataclass(slots=True)
class Message:
    """Message 是会话的基本单元。

    核心不变量：带 tool_calls 的 assistant 消息后必须紧跟 tool_results，
    中间不允许插入其他消息类型，否则 Anthropic/OpenAI 均返回 400。

    content_parts 为多模态扩展（OpenAI scheme only）。非 USER 角色
    使用 content_parts 时，OpenAI adapter 会抛出 ValueError。
    """
    role: Role
    content: str | None = None
    content_parts: list["ContentPart"] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run all multimodal tests**

Run: `uv run pytest tests/test_message_multimodal.py -v`
Expected: all tests pass.

- [ ] **Step 5: Run full test suite to confirm zero regression**

Run: `uv run pytest -v`
Expected: all previously green tests still green.

- [ ] **Step 6: Commit**

```bash
git add src/topsport_agent/types/message.py tests/test_message_multimodal.py
git commit -m "feat(types): Message.content_parts field (backward compatible)"
```

---

## Task 4: OpenAI adapter — USER branch for TextPart + ImagePart(URL)

**Files:**
- Modify: `src/topsport_agent/llm/adapters/openai_chat.py`
- Modify: `tests/test_openai_adapter.py`

- [ ] **Step 1: Inspect existing test file**

Run: `head -40 tests/test_openai_adapter.py`
Note the test helper imports and any existing `_build_request` helpers to reuse.

- [ ] **Step 2: Append failing tests**

Append to `tests/test_openai_adapter.py`:

```python
from pathlib import Path

from topsport_agent.llm.adapters.openai_chat import OpenAIChatAdapter
from topsport_agent.llm.request import LLMRequest
from topsport_agent.types.message import (
    ImagePart,
    Message,
    Role,
    TextPart,
    VideoPart,
    image_bytes,
    image_file,
    image_url,
    video_url,
)


def _req(messages: list[Message]) -> LLMRequest:
    return LLMRequest(model="gpt-4o", messages=messages)


def test_user_message_plain_string_payload_unchanged() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(role=Role.USER, content="hello")
    payload = adapter.build_payload(_req([msg]))
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_user_message_with_text_part_only_emits_array() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(role=Role.USER, content=None, content_parts=[TextPart("hi")])
    payload = adapter.build_payload(_req([msg]))
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]


def test_user_message_with_image_url() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(
        role=Role.USER,
        content=None,
        content_parts=[
            TextPart("what is this?"),
            image_url("https://example.com/a.jpg"),
        ],
    )
    payload = adapter.build_payload(_req([msg]))
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/a.jpg",
            "detail": "auto",
        },
    }


def test_user_message_with_content_and_parts_prepends_text() -> None:
    """When both content and content_parts are set for USER, content
    becomes the first TextPart."""
    adapter = OpenAIChatAdapter()
    msg = Message(
        role=Role.USER,
        content="lead",
        content_parts=[image_url("https://x")],
    )
    payload = adapter.build_payload(_req([msg]))
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "lead"}
    assert content[1]["type"] == "image_url"


def test_user_message_with_image_detail_high() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(
        role=Role.USER,
        content_parts=[image_url("https://x.jpg", detail="high")],
    )
    payload = adapter.build_payload(_req([msg]))
    assert payload["messages"][0]["content"][0]["image_url"]["detail"] == "high"
```

- [ ] **Step 3: Run tests to confirm failure**

Run: `uv run pytest tests/test_openai_adapter.py -v -k "user_message"`
Expected: existing `test_user_message_plain_string_payload_unchanged` may pass if it matches current behavior; new tests FAIL (content array path not implemented).

- [ ] **Step 4: Extend `_convert_messages` USER branch**

In `src/topsport_agent/llm/adapters/openai_chat.py`, add imports at the top:

```python
from ...types.message import (
    ContentPart,
    ImagePart,
    MediaRef,
    Message,
    Role,
    TextPart,
    ToolCall,
    VideoPart,
)
```

(Keep existing import block; replace only the single `Message, Role, ToolCall` import.)

Add a module-level constant + helper right before `class OpenAIChatAdapter`:

```python
import base64

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


def _resolve_media_url(ref: MediaRef) -> str:
    """Normalize a MediaRef to a URL string.

    URL → pass through.
    Path → read bytes, base64, build data URI (infer media_type from suffix).
    Bytes → base64, build data URI (media_type already validated).
    """
    if ref.url is not None:
        return ref.url
    if ref.path is not None:
        suffix = ref.path.suffix.lower()
        media_type = ref.media_type or _EXT_TO_MEDIA.get(suffix)
        if not media_type:
            raise ValueError(
                f"Cannot infer media_type from {ref.path!r}; "
                "pass MediaRef(media_type=...) explicitly"
            )
        raw = ref.path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{media_type};base64,{b64}"
    if ref.data is not None:
        assert ref.media_type is not None  # guaranteed by MediaRef.__post_init__
        b64 = base64.b64encode(ref.data).decode("ascii")
        return f"data:{ref.media_type};base64,{b64}"
    raise ValueError("MediaRef has no url/path/data")  # unreachable


def _encode_content_part(part: ContentPart) -> dict[str, Any]:
    """Render a ContentPart into an OpenAI content block."""
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return {
            "type": "image_url",
            "image_url": {
                "url": _resolve_media_url(part.source),
                "detail": part.detail,
            },
        }
    if isinstance(part, VideoPart):
        return {
            "type": "video_url",
            "video_url": {"url": _resolve_media_url(part.source)},
        }
    raise TypeError(f"Unsupported content part: {type(part).__name__}")
```

Then in `_convert_messages`, replace the USER branch:

```python
if msg.role == Role.USER:
    if msg.content_parts is None:
        converted.append({"role": "user", "content": msg.content or ""})
        continue
    parts_payload: list[dict[str, Any]] = []
    if msg.content:
        parts_payload.append({"type": "text", "text": msg.content})
    for part in msg.content_parts:
        parts_payload.append(_encode_content_part(part))
    converted.append({"role": "user", "content": parts_payload})
    continue
```

- [ ] **Step 5: Run the new tests to confirm pass**

Run: `uv run pytest tests/test_openai_adapter.py -v -k "user_message"`
Expected: all user_message tests pass.

- [ ] **Step 6: Run full openai adapter suite**

Run: `uv run pytest tests/test_openai_adapter.py -v`
Expected: all tests pass (existing + new).

- [ ] **Step 7: Commit**

```bash
git add src/topsport_agent/llm/adapters/openai_chat.py tests/test_openai_adapter.py
git commit -m "feat(adapter): OpenAI user message supports text + image_url parts"
```

---

## Task 5: OpenAI adapter — local file + bytes → data URI

**Files:**
- Modify: `tests/test_openai_adapter.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_openai_adapter.py`:

```python
def test_user_message_with_image_file_auto_base64(tmp_path: Path) -> None:
    adapter = OpenAIChatAdapter()
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    msg = Message(role=Role.USER, content_parts=[image_file(img)])
    payload = adapter.build_payload(_req([msg]))
    url = payload["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_user_message_with_image_bytes_explicit_media_type() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(
        role=Role.USER,
        content_parts=[image_bytes(b"raw-jpeg-data", "image/jpeg")],
    )
    payload = adapter.build_payload(_req([msg]))
    url = payload["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    import base64 as _b64
    assert _b64.b64decode(url.split(",", 1)[1]) == b"raw-jpeg-data"


def test_path_with_unknown_extension_and_no_media_type_raises(
    tmp_path: Path,
) -> None:
    adapter = OpenAIChatAdapter()
    f = tmp_path / "mystery.bin"
    f.write_bytes(b"stuff")
    msg = Message(role=Role.USER, content_parts=[image_file(f)])
    import pytest
    with pytest.raises(ValueError, match="Cannot infer media_type"):
        adapter.build_payload(_req([msg]))
```

- [ ] **Step 2: Run tests to confirm they pass (already implemented in Task 4)**

Run: `uv run pytest tests/test_openai_adapter.py -v -k "image_file or image_bytes or unknown_extension"`
Expected: all pass (Task 4 already wired the three code paths).

- [ ] **Step 3: Commit**

```bash
git add tests/test_openai_adapter.py
git commit -m "test(adapter): cover local file + bytes + unknown extension paths"
```

---

## Task 6: OpenAI adapter — video_url part

**Files:**
- Modify: `tests/test_openai_adapter.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_openai_adapter.py`:

```python
def test_user_message_with_video_url() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(
        role=Role.USER,
        content=None,
        content_parts=[
            TextPart("describe this clip"),
            video_url("https://example.com/clip.mp4"),
        ],
    )
    payload = adapter.build_payload(_req([msg]))
    blocks = payload["messages"][0]["content"]
    assert blocks[1] == {
        "type": "video_url",
        "video_url": {"url": "https://example.com/clip.mp4"},
    }
```

- [ ] **Step 2: Run test to confirm pass (Task 4 already handles VideoPart)**

Run: `uv run pytest tests/test_openai_adapter.py::test_user_message_with_video_url -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_openai_adapter.py
git commit -m "test(adapter): cover video_url content part"
```

---

## Task 7: OpenAI adapter — reject content_parts on non-USER roles

**Files:**
- Modify: `src/topsport_agent/llm/adapters/openai_chat.py`
- Modify: `tests/test_openai_adapter.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_openai_adapter.py`:

```python
def test_assistant_role_with_content_parts_raises() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(
        role=Role.ASSISTANT,
        content_parts=[TextPart("should not be allowed")],
    )
    import pytest
    with pytest.raises(ValueError, match="assistant.*content_parts"):
        adapter.build_payload(_req([msg]))


def test_system_role_with_content_parts_raises() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(
        role=Role.SYSTEM,
        content_parts=[TextPart("nope")],
    )
    import pytest
    with pytest.raises(ValueError, match="system.*content_parts"):
        adapter.build_payload(_req([msg]))


def test_tool_role_with_content_parts_raises() -> None:
    adapter = OpenAIChatAdapter()
    msg = Message(
        role=Role.TOOL,
        content_parts=[TextPart("nope")],
    )
    import pytest
    with pytest.raises(ValueError, match="tool.*content_parts"):
        adapter.build_payload(_req([msg]))
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_openai_adapter.py -v -k "role_with_content_parts"`
Expected: FAIL (no validation yet).

- [ ] **Step 3: Add validation in `_convert_messages`**

In `src/topsport_agent/llm/adapters/openai_chat.py::_convert_messages`, add this as the **very first line inside the `for msg in messages:` loop** (before any role branch):

```python
for msg in messages:
    if msg.content_parts is not None and msg.role != Role.USER:
        raise ValueError(
            f"{msg.role.value} messages do not support content_parts (OpenAI scheme)"
        )
    if msg.role == Role.SYSTEM:
        ...  # existing code
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `uv run pytest tests/test_openai_adapter.py -v -k "role_with_content_parts"`
Expected: 3 tests pass.

- [ ] **Step 5: Run full adapter suite**

Run: `uv run pytest tests/test_openai_adapter.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/topsport_agent/llm/adapters/openai_chat.py tests/test_openai_adapter.py
git commit -m "feat(adapter): reject content_parts on non-USER roles"
```

---

## Task 8: Image generation data types

**Files:**
- Create: `src/topsport_agent/llm/image_generation.py`
- Create: `tests/test_image_generation.py`

- [ ] **Step 1: Create failing test**

Create `tests/test_image_generation.py`:

```python
from __future__ import annotations

import pytest

from topsport_agent.llm.image_generation import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResponse,
)


def test_image_generation_request_defaults() -> None:
    req = ImageGenerationRequest(prompt="a cat", model="dall-e-3")
    assert req.prompt == "a cat"
    assert req.model == "dall-e-3"
    assert req.size is None
    assert req.quality is None
    assert req.style is None
    assert req.n == 1
    assert req.response_format == "url"
    assert req.provider_options == {}


def test_image_generation_request_accepts_all_options() -> None:
    req = ImageGenerationRequest(
        prompt="cat",
        model="dall-e-3",
        size="1024x1024",
        quality="hd",
        style="vivid",
        n=2,
        response_format="b64_json",
        provider_options={"user": "u1"},
    )
    assert req.size == "1024x1024"
    assert req.quality == "hd"
    assert req.response_format == "b64_json"
    assert req.provider_options == {"user": "u1"}


def test_generated_image_defaults() -> None:
    img = GeneratedImage()
    assert img.url is None
    assert img.b64_json is None
    assert img.revised_prompt is None


def test_image_generation_response_holds_images() -> None:
    resp = ImageGenerationResponse(
        images=[GeneratedImage(url="https://x/out.png")],
    )
    assert len(resp.images) == 1
    assert resp.usage == {}
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_image_generation.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Create `llm/image_generation.py` with types only**

Create `src/topsport_agent/llm/image_generation.py`:

```python
from __future__ import annotations

import base64
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


@dataclass(slots=True)
class ImageGenerationRequest:
    """Image generation request aligned with OpenAI /v1/images/generations."""
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
    """Single generated image. Exactly one of url/b64_json is set by the
    provider; revised_prompt is DALL-E-3-specific."""
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None

    async def save(
        self,
        path: str | Path,
        *,
        http_client: Any | None = None,
    ) -> Path:
        """Save this image to disk. b64_json is decoded; url is downloaded."""
        target = Path(path)
        if self.b64_json is not None:
            target.write_bytes(base64.b64decode(self.b64_json))
            return target
        if self.url is not None:
            mod_name = "httpx"
            httpx_mod = importlib.import_module(mod_name)
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


@dataclass(slots=True)
class ImageGenerationResponse:
    """Response from OpenAIImageGenerationClient.generate."""
    images: list[GeneratedImage]
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
```

- [ ] **Step 4: Run tests to confirm pass**

Run: `uv run pytest tests/test_image_generation.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/llm/image_generation.py tests/test_image_generation.py
git commit -m "feat(image-gen): request/response/image dataclasses"
```

---

## Task 9: OpenAIImageGenerationClient — generate()

**Files:**
- Modify: `src/topsport_agent/llm/image_generation.py`
- Modify: `tests/test_image_generation.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_image_generation.py`:

```python
from types import SimpleNamespace

import pytest

from topsport_agent.llm.image_generation import OpenAIImageGenerationClient


class _CapturingImages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.captured: dict[str, Any] | None = None

    async def generate(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        return self.result


class _MockClient:
    def __init__(self, result: Any) -> None:
        self.images = _CapturingImages(result)


def _result(url: str = "https://example.com/out.png") -> Any:
    return SimpleNamespace(
        data=[SimpleNamespace(url=url, b64_json=None, revised_prompt="refined")],
        usage=None,
    )


def test_client_requires_client_or_factory() -> None:
    with pytest.raises(ValueError, match="client.*factory"):
        OpenAIImageGenerationClient()  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_generate_builds_kwargs_and_parses_response() -> None:
    mock = _MockClient(_result())
    client = OpenAIImageGenerationClient(client=mock)
    resp = await client.generate(
        ImageGenerationRequest(
            prompt="cat", model="dall-e-3", size="1024x1024", quality="hd",
        )
    )
    assert mock.images.captured == {
        "model": "dall-e-3",
        "prompt": "cat",
        "n": 1,
        "response_format": "url",
        "size": "1024x1024",
        "quality": "hd",
    }
    assert resp.images[0].url == "https://example.com/out.png"
    assert resp.images[0].revised_prompt == "refined"


@pytest.mark.asyncio
async def test_provider_options_override_kwargs() -> None:
    mock = _MockClient(_result())
    client = OpenAIImageGenerationClient(client=mock)
    await client.generate(
        ImageGenerationRequest(
            prompt="x", model="dall-e-3",
            provider_options={"model": "overridden", "user": "u1"},
        )
    )
    assert mock.images.captured is not None
    assert mock.images.captured["model"] == "overridden"
    assert mock.images.captured["user"] == "u1"


@pytest.mark.asyncio
async def test_lazy_factory_creates_client_once() -> None:
    calls = {"n": 0}

    def _factory() -> Any:
        calls["n"] += 1
        return _MockClient(_result())

    client = OpenAIImageGenerationClient(client_factory=_factory)
    await client.generate(ImageGenerationRequest(prompt="a", model="m"))
    await client.generate(ImageGenerationRequest(prompt="b", model="m"))
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_parse_response_handles_empty_data_list() -> None:
    mock = _MockClient(SimpleNamespace(data=[], usage=None))
    client = OpenAIImageGenerationClient(client=mock)
    resp = await client.generate(ImageGenerationRequest(prompt="x", model="m"))
    assert resp.images == []


@pytest.mark.asyncio
async def test_parse_response_extracts_usage_when_present() -> None:
    mock = _MockClient(
        SimpleNamespace(
            data=[SimpleNamespace(url="u", b64_json=None, revised_prompt=None)],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=20, total_tokens=30
            ),
        )
    )
    client = OpenAIImageGenerationClient(client=mock)
    resp = await client.generate(ImageGenerationRequest(prompt="x", model="m"))
    assert resp.usage == {
        "input_tokens": 10, "output_tokens": 20, "total_tokens": 30
    }
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `uv run pytest tests/test_image_generation.py -v`
Expected: FAIL — `OpenAIImageGenerationClient` not defined.

- [ ] **Step 3: Add the client to `llm/image_generation.py`**

Append to `src/topsport_agent/llm/image_generation.py`:

```python
class OpenAIImageGenerationClient:
    """Synchronous-request client for OpenAI /v1/images/generations.

    Construct with an injected `client` (for tests) or a `client_factory`
    (for production, lazy-imports openai.AsyncOpenAI).
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        default_model: str | None = None,
    ) -> None:
        if client is None and client_factory is None:
            raise ValueError(
                "Either `client` or `client_factory` is required"
            )
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
        if req.size:
            kwargs["size"] = req.size
        if req.quality:
            kwargs["quality"] = req.quality
        if req.style:
            kwargs["style"] = req.style
        kwargs.update(req.provider_options)
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

- [ ] **Step 4: Run tests to confirm pass**

Run: `uv run pytest tests/test_image_generation.py -v`
Expected: all tests pass (4 existing + 6 new = 10).

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/llm/image_generation.py tests/test_image_generation.py
git commit -m "feat(image-gen): OpenAIImageGenerationClient with injected client"
```

---

## Task 10: GeneratedImage.save() — both paths

**Files:**
- Modify: `tests/test_image_generation.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_image_generation.py`:

```python
@pytest.mark.asyncio
async def test_save_b64_writes_decoded_bytes(tmp_path) -> None:
    import base64 as _b64
    payload = _b64.b64encode(b"fake-png-bytes").decode("ascii")
    img = GeneratedImage(b64_json=payload)
    target = await img.save(tmp_path / "out.png")
    assert target.read_bytes() == b"fake-png-bytes"


@pytest.mark.asyncio
async def test_save_url_uses_injected_http_client(tmp_path) -> None:
    class _MockResp:
        def __init__(self, content: bytes) -> None:
            self.content = content
        def raise_for_status(self) -> None:
            pass

    class _MockHttp:
        def __init__(self) -> None:
            self.calls: list[str] = []
        async def get(self, url: str) -> _MockResp:
            self.calls.append(url)
            return _MockResp(b"downloaded")

    http = _MockHttp()
    img = GeneratedImage(url="https://example.com/x.png")
    target = await img.save(tmp_path / "out.png", http_client=http)
    assert http.calls == ["https://example.com/x.png"]
    assert target.read_bytes() == b"downloaded"


@pytest.mark.asyncio
async def test_save_raises_when_neither_url_nor_b64(tmp_path) -> None:
    img = GeneratedImage()
    with pytest.raises(ValueError, match="neither url nor b64_json"):
        await img.save(tmp_path / "x.png")
```

- [ ] **Step 2: Run tests to confirm pass (save already implemented in Task 8)**

Run: `uv run pytest tests/test_image_generation.py -v -k "save_"`
Expected: 3 tests pass.

- [ ] **Step 3: Verify module stays importable without httpx**

Run:
```bash
uv run python -c "from topsport_agent.llm.image_generation import GeneratedImage, OpenAIImageGenerationClient; print('ok')"
```
Expected output: `ok`.
(The `httpx` import is inside `save()`, not at module level.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_image_generation.py
git commit -m "test(image-gen): cover GeneratedImage.save for b64 and url paths"
```

---

## Task 11: Agent.run accepts list[ContentPart] and Message

**Files:**
- Modify: `src/topsport_agent/agent/base.py`
- Create: `tests/test_agent_multimodal.py`

- [ ] **Step 1: Inspect Agent constructor and existing test style**

Run: `grep -n "^class Agent\|def __init__\|def run\|def new_session" src/topsport_agent/agent/base.py | head -10`

Also run: `ls tests | grep agent`

Look at one existing agent test to reuse its scaffolding (e.g., how it builds a mock `provider` + `Engine`).

- [ ] **Step 2: Create failing test file**

Create `tests/test_agent_multimodal.py`:

```python
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

from topsport_agent.agent.base import Agent, AgentConfig
from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.types.message import (
    ContentPart,
    ImagePart,
    Message,
    Role,
    TextPart,
    image_url,
)


class _StubProvider:
    name = "stub"

    async def complete(self, request: Any) -> Any:
        return SimpleNamespace(
            text="ok", tool_calls=[], finish_reason="stop",
            usage={}, response_metadata=None, raw=None,
        )


def _make_agent() -> Agent:
    provider = _StubProvider()
    engine = Engine(
        provider=provider,
        tools=[],
        config=EngineConfig(model="test-model"),
    )
    return Agent(provider=provider, config=AgentConfig(), engine=engine)


@pytest.mark.asyncio
async def test_run_accepts_string_as_before() -> None:
    agent = _make_agent()
    session = agent.new_session()
    async for _ in agent.run("hello", session):
        break
    assert session.messages[0].role == Role.USER
    assert session.messages[0].content == "hello"
    assert session.messages[0].content_parts is None


@pytest.mark.asyncio
async def test_run_accepts_content_parts_list() -> None:
    agent = _make_agent()
    session = agent.new_session()
    parts: list[ContentPart] = [
        TextPart("describe"),
        image_url("https://example.com/a.jpg"),
    ]
    async for _ in agent.run(parts, session):
        break
    msg = session.messages[0]
    assert msg.role == Role.USER
    assert msg.content is None
    assert msg.content_parts == parts


@pytest.mark.asyncio
async def test_run_accepts_prebuilt_message() -> None:
    agent = _make_agent()
    session = agent.new_session()
    msg = Message(
        role=Role.USER,
        content_parts=[TextPart("hi")],
        extra={"uid": "u1"},
    )
    async for _ in agent.run(msg, session):
        break
    assert session.messages[0] is msg
    assert session.messages[0].extra == {"uid": "u1"}
```

- [ ] **Step 3: Run to confirm failure**

Run: `uv run pytest tests/test_agent_multimodal.py -v`
Expected: FAIL — `Agent.run` rejects list / Message.

- [ ] **Step 4: Extend `Agent.run` signature**

Modify `src/topsport_agent/agent/base.py::Agent.run`:

```python
async def run(
    self,
    user_input: str | list["ContentPart"] | Message,
    session: Session,
) -> AsyncIterator[Event]:
    """附加一条用户消息到 session，驱动 engine 跑一轮推理。

    user_input 接受三种形态：
      - str: 向后兼容的纯文本
      - list[ContentPart]: 多模态 parts，自动包装为 USER 消息
      - Message: 用户构造好的完整消息，按角色/字段原样使用
    """
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

Also add `ContentPart` to the imports at the top of the file:

```python
from ..types.message import ContentPart, Message, Role
```

(Extend the existing `Message, Role` import; keep everything else unchanged.)

- [ ] **Step 5: Run tests to confirm pass**

Run: `uv run pytest tests/test_agent_multimodal.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Run full test suite to confirm no regression**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/topsport_agent/agent/base.py tests/test_agent_multimodal.py
git commit -m "feat(agent): run() accepts str | list[ContentPart] | Message"
```

---

## Task 12: AgentConfig.image_generator + Agent.generate_image()

**Files:**
- Modify: `src/topsport_agent/agent/base.py`
- Modify: `tests/test_agent_multimodal.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_agent_multimodal.py`:

```python
from topsport_agent.llm.image_generation import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResponse,
    OpenAIImageGenerationClient,
)


class _RecordingImageGen:
    """Test double for OpenAIImageGenerationClient."""

    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model
        self.last_request: ImageGenerationRequest | None = None

    async def generate(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        self.last_request = request
        return ImageGenerationResponse(
            images=[GeneratedImage(url="https://example.com/out.png")],
        )


@pytest.mark.asyncio
async def test_generate_image_raises_when_not_configured() -> None:
    agent = _make_agent()
    with pytest.raises(RuntimeError, match="No image_generator"):
        await agent.generate_image("anything")


@pytest.mark.asyncio
async def test_generate_image_delegates_and_uses_default_model() -> None:
    gen = _RecordingImageGen(default_model="dall-e-3")
    agent = Agent(
        provider=_StubProvider(),
        config=AgentConfig(image_generator=gen),  # type: ignore[arg-type]
        engine=Engine(provider=_StubProvider(), tools=[], config=EngineConfig(model="test-model")),
        image_generator=gen,  # type: ignore[arg-type]
    )
    resp = await agent.generate_image("cat", size="512x512")
    assert gen.last_request is not None
    assert gen.last_request.prompt == "cat"
    assert gen.last_request.size == "512x512"
    assert gen.last_request.model == "dall-e-3"
    assert resp.images[0].url == "https://example.com/out.png"


@pytest.mark.asyncio
async def test_generate_image_explicit_model_overrides_default() -> None:
    gen = _RecordingImageGen(default_model="dall-e-3")
    agent = Agent(
        provider=_StubProvider(),
        config=AgentConfig(image_generator=gen),  # type: ignore[arg-type]
        engine=Engine(provider=_StubProvider(), tools=[], config=EngineConfig(model="test-model")),
        image_generator=gen,  # type: ignore[arg-type]
    )
    await agent.generate_image("cat", model="gpt-image-1")
    assert gen.last_request is not None
    assert gen.last_request.model == "gpt-image-1"


@pytest.mark.asyncio
async def test_generate_image_requires_model_when_no_default() -> None:
    gen = _RecordingImageGen(default_model=None)
    agent = Agent(
        provider=_StubProvider(),
        config=AgentConfig(image_generator=gen),  # type: ignore[arg-type]
        engine=Engine(provider=_StubProvider(), tools=[], config=EngineConfig(model="test-model")),
        image_generator=gen,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="model required"):
        await agent.generate_image("cat")
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_agent_multimodal.py -v -k "generate_image"`
Expected: FAIL — `Agent.generate_image` does not exist.

- [ ] **Step 3: Extend AgentConfig**

In `src/topsport_agent/agent/base.py`, add to `AgentConfig`:

```python
@dataclass
class AgentConfig:
    # ...existing fields...
    image_generator: "OpenAIImageGenerationClient | None" = None
```

Add a TYPE_CHECKING import at the top of `base.py` (inside the existing `TYPE_CHECKING` block):

```python
if TYPE_CHECKING:
    # existing imports
    from ..llm.image_generation import (
        ImageGenerationResponse,
        OpenAIImageGenerationClient,
    )
```

- [ ] **Step 4: Extend Agent.__init__ and add generate_image**

In `Agent.__init__`, add keyword-only parameter:

```python
def __init__(
    self,
    provider: LLMProvider,
    config: AgentConfig,
    *,
    engine: Engine,
    cleanup_callbacks: list[Callable[[], Awaitable[None]]] | None = None,
    skill_registry: SkillRegistry | None = None,
    plugin_manager: PluginManager | None = None,
    capability_bundle: dict[str, Any] | None = None,
    image_generator: "OpenAIImageGenerationClient | None" = None,
) -> None:
    # ...existing assignments...
    self._image_generator = image_generator
    self._capability_bundle = capability_bundle or {}
```

Add the method after `run()`:

```python
async def generate_image(
    self,
    prompt: str,
    *,
    model: str | None = None,
    **kwargs: Any,
) -> "ImageGenerationResponse":
    """Generate an image via the configured image_generator.

    Raises RuntimeError if no image_generator was wired into this Agent.
    """
    if self._image_generator is None:
        raise RuntimeError(
            "No image_generator configured; pass image_generator= "
            "to Agent(), or construct OpenAIImageGenerationClient directly."
        )
    from ..llm.image_generation import ImageGenerationRequest  # lazy

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

- [ ] **Step 5: Run tests to confirm pass**

Run: `uv run pytest tests/test_agent_multimodal.py -v -k "generate_image"`
Expected: 4 tests pass.

- [ ] **Step 6: Run full agent multimodal suite**

Run: `uv run pytest tests/test_agent_multimodal.py -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/topsport_agent/agent/base.py tests/test_agent_multimodal.py
git commit -m "feat(agent): generate_image convenience + AgentConfig.image_generator"
```

---

## Task 13: Agent.from_config forwards image_generator

**Files:**
- Modify: `src/topsport_agent/agent/base.py`
- Modify: `tests/test_agent_multimodal.py`

- [ ] **Step 1: Append failing test**

Append to `tests/test_agent_multimodal.py`:

```python
@pytest.mark.asyncio
async def test_from_config_forwards_image_generator() -> None:
    gen = _RecordingImageGen(default_model="dall-e-3")
    provider = _StubProvider()
    config = AgentConfig(image_generator=gen)  # type: ignore[arg-type]
    agent = Agent.from_config(provider=provider, config=config)
    resp = await agent.generate_image("cat")
    assert resp.images[0].url == "https://example.com/out.png"
    assert gen.last_request is not None
    assert gen.last_request.model == "dall-e-3"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_agent_multimodal.py::test_from_config_forwards_image_generator -v`
Expected: FAIL — `from_config` does not pass the generator.

- [ ] **Step 3: Wire `from_config`**

Find the `return cls(...)` call at the end of `Agent.from_config` in `src/topsport_agent/agent/base.py` and add `image_generator=config.image_generator`:

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

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_agent_multimodal.py::test_from_config_forwards_image_generator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/agent/base.py tests/test_agent_multimodal.py
git commit -m "feat(agent): from_config forwards image_generator to Agent"
```

---

## Task 14: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Locate the right section**

Run: `grep -n "^##" README.md`

Pick a position after existing "Usage" / "Quick start" section, or append a new "Multimodal Input & Image Generation" section near the end of the usage area.

- [ ] **Step 2: Add the section**

Insert the following content (adjust indentation to match surrounding headings):

````markdown
## Multimodal Input & Image Generation

This section applies to OpenAI-compatible endpoints only. The Anthropic
adapter does not support multimodal content in this release.

### Analyzing images and videos

```python
from topsport_agent.types.message import (
    TextPart, image_url, image_file, video_url,
)

session = agent.new_session()
async for event in agent.run(
    [
        TextPart("Compare these two images"),
        image_url("https://cdn.example.com/a.jpg"),
        image_file("/tmp/b.png", detail="high"),
    ],
    session,
):
    ...
```

Videos use the `video_url` content part (requires a Qwen-VL-compatible
endpoint):

```python
async for event in agent.run(
    [TextPart("describe this clip"), video_url("https://.../clip.mp4")],
    session,
):
    ...
```

Local files are read and base64-encoded into data URIs automatically.
For files with uncommon extensions, pass an explicit `media_type`:

```python
from topsport_agent.types.message import MediaRef, ImagePart

ImagePart(
    source=MediaRef(path=Path("/tmp/x.heic"), media_type="image/heic"),
)
```

### Generating images

Image generation is a separate, synchronous API. It does not go through
the LLM loop.

```python
from topsport_agent.llm.image_generation import OpenAIImageGenerationClient
from topsport_agent.agent.base import Agent, AgentConfig

def _openai_factory():
    import openai
    return openai.AsyncOpenAI(
        api_key=os.environ["OPENAI_COMPAT_KEY"],
        base_url=os.environ["OPENAI_COMPAT_BASE_URL"],
    )

image_gen = OpenAIImageGenerationClient(
    client_factory=_openai_factory,
    default_model="dall-e-3",
)

agent = Agent.from_config(
    provider=my_provider,
    config=AgentConfig(image_generator=image_gen),
)

resp = await agent.generate_image("a cyberpunk cat", size="1024x1024")
await resp.images[0].save("./cat.png")
```

If the underlying endpoint does not support `/v1/images/generations`
(e.g., DashScope's compat layer), the provider's HTTP error is
surfaced unchanged. Configure a proxy like one-api/new-api to bridge
to async providers.
````

- [ ] **Step 3: Verify markdownlint compliance**

Run: `npx -y markdownlint-cli README.md 2>&1 | head -20` (or if the project uses a different lint tool, use that; otherwise visually check headings/lists are consistent).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): multimodal input + image generation usage"
```

---

## Task 15: End-to-end verification + learnings sweep

**Files:**
- (Verify-only + maybe `.learnings/LEARNINGS.md`)

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest -v`
Expected: all green, no new xfail/skip.

- [ ] **Step 2: Verify no top-level openai/httpx imports in tests**

Run:
```bash
grep -l "^import openai\|^from openai\|^import httpx\|^from httpx" tests/ -r || echo "no top-level optional imports"
```
Expected output: `no top-level optional imports`.

- [ ] **Step 3: Verify module-level optional-dep clean import**

Run:
```bash
uv run python -c "
from topsport_agent.types.message import MediaRef, ImagePart, VideoPart, TextPart, image_url
from topsport_agent.llm.image_generation import OpenAIImageGenerationClient, GeneratedImage
from topsport_agent.llm.adapters.openai_chat import OpenAIChatAdapter
from topsport_agent.agent.base import Agent, AgentConfig
print('all modules importable')
"
```
Expected output: `all modules importable`.

- [ ] **Step 4: Capture any non-obvious issues to `.learnings/`**

If during implementation any of these surfaced (review commit messages / test failures):

- Surprising behavior in the OpenAI SDK's `images.generate` parameters
- Unexpected `httpx` import chain
- Pyright false-positives in `content_parts: list["ContentPart"]` self-ref

…then add an entry to `.learnings/LEARNINGS.md` in the existing Context / Learned / Evidence format.

If nothing surprising emerged, skip this step.

- [ ] **Step 5: Final commit (if learnings updated)**

```bash
git add .learnings/LEARNINGS.md
git commit -m "docs(learnings): multimodal implementation notes"
```

---

## Done Checklist

- [ ] `uv run pytest -v` is fully green
- [ ] No top-level `import openai` / `import httpx` in `tests/`
- [ ] `README.md` has the new section, markdownlint-clean
- [ ] `tests/test_openai_adapter.py` existing cases unchanged
- [ ] `tests/test_anthropic_adapter.py` unchanged
- [ ] 14 commits on the branch (or single merge commit to `main`)
- [ ] Pyright reports no new errors (`uv run pyright src tests` if configured)
