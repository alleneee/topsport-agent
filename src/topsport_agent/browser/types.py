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
    """A single interactive element in the page snapshot.

    nth 仅在同 (role, name) 出现多次时填写，用于 Playwright get_by_role(...).nth(n)
    消歧；唯一元素上保持 None，走 .first 分支与原行为兼容。
    """

    ref: str
    role: str
    name: str
    nth: int | None = None
    tag: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PageSnapshot:
    """Structured snapshot of interactive elements on the current page."""

    url: str
    title: str
    entries: list[SnapshotEntry] = field(default_factory=list)
    frame_selector: str = ""

    def to_text(self) -> str:
        lines = [f"URL: {self.url}", f"Title: {self.title}"]
        if self.frame_selector:
            lines.append(f"Frame: {self.frame_selector}")
        lines.append("")
        for e in self.entries:
            name_part = f' "{e.name}"' if e.name else ""
            nth_part = f" [nth={e.nth}]" if e.nth is not None else ""
            lines.append(f"  {e.ref} [{e.role}]{name_part}{nth_part}")
        return "\n".join(lines)
