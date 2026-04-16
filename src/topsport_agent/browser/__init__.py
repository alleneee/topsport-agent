"""Built-in browser control module."""

from .client import BrowserClient, PageFactory
from .tools import BrowserToolSource
from .types import BrowserConfig, PageSnapshot, SnapshotEntry

__all__ = [
    "BrowserClient",
    "BrowserConfig",
    "BrowserToolSource",
    "PageFactory",
    "PageSnapshot",
    "SnapshotEntry",
]
