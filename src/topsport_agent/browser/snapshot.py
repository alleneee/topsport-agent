"""Accessibility tree parsing and @ref assignment."""

from __future__ import annotations

from typing import Any

from .types import PageSnapshot, SnapshotEntry

INTERACTIVE_ROLES = frozenset({
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "menuitem",
    "tab",
    "switch",
    "searchbox",
    "slider",
    "spinbutton",
    "option",
    "menuitemcheckbox",
    "menuitemradio",
})


async def take_snapshot(page: Any) -> PageSnapshot:
    """Extract interactive elements from accessibility tree, assign sequential @refs."""
    tree = await page.accessibility_snapshot()
    url = page.url
    title = await page.title()

    entries: list[SnapshotEntry] = []
    counter = 0

    def walk(node: dict[str, Any]) -> None:
        nonlocal counter
        role = node.get("role", "")
        if role in INTERACTIVE_ROLES:
            counter += 1
            entries.append(
                SnapshotEntry(
                    ref=f"@e{counter}",
                    role=role,
                    name=node.get("name", ""),
                )
            )
        for child in node.get("children", []):
            walk(child)

    if tree:
        walk(tree)

    return PageSnapshot(url=url, title=title, entries=entries)


def build_ref_map(snapshot: PageSnapshot) -> dict[str, tuple[str, str]]:
    """Build {ref: (role, name)} mapping for locator resolution."""
    return {e.ref: (e.role, e.name) for e in snapshot.entries}
