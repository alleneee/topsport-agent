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
    with pytest.raises(Exception):
        ref.url = "https://y"  # type: ignore[misc]


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
