"""Accessibility tree parsing and @ref assignment.

Uses Playwright's ``page.locator("body").aria_snapshot()`` which returns a
YAML-formatted string of the accessibility tree. Each line looks like::

    - role "name" [attr=value]
    - role "name":
      - childrole "childname"

当同 (role, name) 在一页出现多次时，为避免 get_by_role(role, name).first 永远指向
第一个元素而误操作，对重复项附带 ``nth=i`` 序号；唯一元素仍保持 ``nth=None``，
让常规路径继续命中，保持既有测试和 LLM 用法不变。
"""

from __future__ import annotations

import re
from typing import Any

from .types import PageSnapshot, SnapshotEntry

INTERACTIVE_ROLES = frozenset({
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "treeitem",
})

# 放宽后兼容三种 suffix：空、`[disabled]`、`[ref=e1] [nth=0]:` 等任意组合，
# 同时接受 `- /button` 形式的 ARIA 关闭标记（后续根据前缀跳过）。
_LINE_RE = re.compile(r'^-\s+(/?\w+)(?:\s+"([^"]*)")?(.*)$')


def _parse_aria_yaml(yaml_text: str) -> list[SnapshotEntry]:
    """Parse aria_snapshot YAML output, extract interactive elements with @refs."""
    entries: list[SnapshotEntry] = []
    dedup: dict[tuple[str, str], list[SnapshotEntry]] = {}
    counter = 0

    for line in yaml_text.splitlines():
        stripped = line.lstrip()
        m = _LINE_RE.match(stripped)
        if not m:
            continue
        role = m.group(1)
        if role.startswith("/"):
            continue
        if role not in INTERACTIVE_ROLES:
            continue
        name = m.group(2) or ""
        counter += 1
        entry = SnapshotEntry(ref=f"@e{counter}", role=role, name=name)
        entries.append(entry)
        dedup.setdefault((role, name), []).append(entry)

    for group in dedup.values():
        if len(group) > 1:
            for idx, entry in enumerate(group):
                entry.nth = idx

    return entries


async def take_snapshot(page: Any, frame_selector: str = "") -> PageSnapshot:
    """Extract interactive elements from accessibility tree, assign sequential @refs.

    frame_selector 非空时 snapshot 作用域切换到对应 iframe；url/title 仍取
    顶层 page，它是 LLM 用来核对上下文的稳定锚点。
    """
    root = page.frame_locator(frame_selector) if frame_selector else page
    yaml_text = await root.locator("body").aria_snapshot()
    url = page.url
    title = await page.title()

    entries = _parse_aria_yaml(yaml_text) if yaml_text else []
    return PageSnapshot(
        url=url,
        title=title,
        entries=entries,
        frame_selector=frame_selector,
    )


def build_ref_map(snapshot: PageSnapshot) -> dict[str, tuple[str, str, int | None]]:
    """Build {ref: (role, name, nth)} mapping for locator resolution."""
    return {e.ref: (e.role, e.name, e.nth) for e in snapshot.entries}
