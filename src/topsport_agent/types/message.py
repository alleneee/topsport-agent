from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


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


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    """ToolCall 是 LLM 发出的工具调用指令，arguments 已解析为 dict（OpenAI 适配器负责 JSON 反序列化）。"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    """ToolResult 是工具执行后的应答。is_error 仅标记引擎层面的失败，MCP 语义错误通过 output 内容区分。"""
    call_id: str
    output: Any
    is_error: bool = False


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
    content_parts: list[ContentPart] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
