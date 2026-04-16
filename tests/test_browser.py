from __future__ import annotations

from topsport_agent.browser.types import BrowserConfig, PageSnapshot, SnapshotEntry
from topsport_agent.browser.snapshot import (
    INTERACTIVE_ROLES,
    build_ref_map,
    take_snapshot,
)


class TestBrowserConfig:
    def test_defaults(self):
        cfg = BrowserConfig()
        assert cfg.headless is True
        assert cfg.viewport_width == 1280
        assert cfg.viewport_height == 720
        assert cfg.default_timeout == 30.0

    def test_custom(self):
        cfg = BrowserConfig(headless=False, viewport_width=800, viewport_height=600, default_timeout=10.0)
        assert cfg.headless is False
        assert cfg.viewport_width == 800


class TestSnapshotEntry:
    def test_fields(self):
        e = SnapshotEntry(ref="@e1", role="button", name="Submit")
        assert e.ref == "@e1"
        assert e.role == "button"
        assert e.name == "Submit"
        assert e.tag == ""
        assert e.attributes == {}


class TestPageSnapshot:
    def test_to_text_empty(self):
        snap = PageSnapshot(url="https://example.com", title="Example")
        text = snap.to_text()
        assert "URL: https://example.com" in text
        assert "Title: Example" in text

    def test_to_text_with_entries(self):
        snap = PageSnapshot(
            url="https://example.com",
            title="Example",
            entries=[
                SnapshotEntry(ref="@e1", role="textbox", name="Email"),
                SnapshotEntry(ref="@e2", role="button", name="Submit"),
            ],
        )
        text = snap.to_text()
        assert '@e1 [textbox] "Email"' in text
        assert '@e2 [button] "Submit"' in text

    def test_to_text_entry_without_name(self):
        snap = PageSnapshot(
            url="https://x.com",
            title="X",
            entries=[SnapshotEntry(ref="@e1", role="link", name="")],
        )
        text = snap.to_text()
        assert "@e1 [link]" in text
        assert '""' not in text


def _make_tree(nodes: list[dict]) -> dict:
    """Helper: build a minimal accessibility tree dict."""
    return {"role": "WebArea", "name": "", "children": nodes}


class TestTakeSnapshot:
    async def test_extracts_interactive_elements(self):
        tree = _make_tree([
            {"role": "textbox", "name": "Email"},
            {"role": "textbox", "name": "Password"},
            {"role": "button", "name": "Login"},
        ])

        class FakePage:
            url = "https://example.com/login"
            async def title(self): return "Login"
            async def accessibility_snapshot(self): return tree

        snap = await take_snapshot(FakePage())
        assert len(snap.entries) == 3
        assert snap.entries[0].ref == "@e1"
        assert snap.entries[0].role == "textbox"
        assert snap.entries[0].name == "Email"
        assert snap.entries[2].ref == "@e3"
        assert snap.entries[2].role == "button"

    async def test_ignores_non_interactive_roles(self):
        tree = _make_tree([
            {"role": "heading", "name": "Welcome"},
            {"role": "text", "name": "some text"},
            {"role": "button", "name": "Click me"},
        ])

        class FakePage:
            url = "https://example.com"
            async def title(self): return "Test"
            async def accessibility_snapshot(self): return tree

        snap = await take_snapshot(FakePage())
        assert len(snap.entries) == 1
        assert snap.entries[0].role == "button"

    async def test_nested_children(self):
        tree = _make_tree([
            {
                "role": "navigation",
                "name": "nav",
                "children": [
                    {"role": "link", "name": "Home"},
                    {"role": "link", "name": "About"},
                ],
            },
        ])

        class FakePage:
            url = "https://example.com"
            async def title(self): return "Test"
            async def accessibility_snapshot(self): return tree

        snap = await take_snapshot(FakePage())
        assert len(snap.entries) == 2
        assert snap.entries[0].ref == "@e1"
        assert snap.entries[1].ref == "@e2"

    async def test_empty_tree(self):
        class FakePage:
            url = "https://example.com"
            async def title(self): return "Empty"
            async def accessibility_snapshot(self): return None

        snap = await take_snapshot(FakePage())
        assert snap.entries == []
        assert snap.url == "https://example.com"

    async def test_sequential_numbering(self):
        tree = _make_tree([
            {"role": "link", "name": "A"},
            {"role": "heading", "name": "skip"},
            {"role": "link", "name": "B"},
            {"role": "textbox", "name": "C"},
        ])

        class FakePage:
            url = "https://example.com"
            async def title(self): return "Test"
            async def accessibility_snapshot(self): return tree

        snap = await take_snapshot(FakePage())
        refs = [e.ref for e in snap.entries]
        assert refs == ["@e1", "@e2", "@e3"]


class TestBuildRefMap:
    def test_builds_map(self):
        snap = PageSnapshot(
            url="https://example.com",
            title="Test",
            entries=[
                SnapshotEntry(ref="@e1", role="button", name="Submit"),
                SnapshotEntry(ref="@e2", role="textbox", name="Email"),
            ],
        )
        ref_map = build_ref_map(snap)
        assert ref_map == {
            "@e1": ("button", "Submit"),
            "@e2": ("textbox", "Email"),
        }
