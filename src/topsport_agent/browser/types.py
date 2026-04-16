from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BrowserConfig:
    """Browser launch configuration."""

    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    default_timeout: float = 30.0


@dataclass(slots=True)
class SnapshotEntry:
    """A single interactive element in the page snapshot."""

    ref: str
    role: str
    name: str
    tag: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PageSnapshot:
    """Structured snapshot of interactive elements on the current page."""

    url: str
    title: str
    entries: list[SnapshotEntry] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"URL: {self.url}", f"Title: {self.title}", ""]
        for e in self.entries:
            name_part = f' "{e.name}"' if e.name else ""
            lines.append(f"  {e.ref} [{e.role}]{name_part}")
        return "\n".join(lines)
