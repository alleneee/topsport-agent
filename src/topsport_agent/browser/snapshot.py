"""Accessibility tree parsing and @ref assignment.

Uses Playwright's ``page.locator("body").aria_snapshot()`` which returns a
YAML-formatted string of the accessibility tree. Each line looks like::

    - role "name" [attr=value]
    - role "name":
      - childrole "childname"
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

# Matches lines like: - button "Submit" [disabled]
# Groups: role, optional quoted name
_LINE_RE = re.compile(r"^-\s+(\w+)(?:\s+\"([^\"]*)\")?\s*(?:\[.*\])?\s*:?\s*$")


def _parse_aria_yaml(yaml_text: str) -> list[SnapshotEntry]:
    """Parse aria_snapshot YAML output, extract interactive elements with @refs."""
    entries: list[SnapshotEntry] = []
    counter = 0

    for line in yaml_text.splitlines():
        stripped = line.lstrip()
        m = _LINE_RE.match(stripped)
        if not m:
            continue
        role = m.group(1)
        name = m.group(2) or ""
        if role in INTERACTIVE_ROLES:
            counter += 1
            entries.append(SnapshotEntry(ref=f"@e{counter}", role=role, name=name))

    return entries


async def take_snapshot(page: Any) -> PageSnapshot:
    """Extract interactive elements from accessibility tree, assign sequential @refs."""
    yaml_text = await page.locator("body").aria_snapshot()
    url = page.url
    title = await page.title()

    entries = _parse_aria_yaml(yaml_text) if yaml_text else []
    return PageSnapshot(url=url, title=title, entries=entries)


def build_ref_map(snapshot: PageSnapshot) -> dict[str, tuple[str, str]]:
    """Build {ref: (role, name)} mapping for locator resolution."""
    return {e.ref: (e.role, e.name) for e in snapshot.entries}
