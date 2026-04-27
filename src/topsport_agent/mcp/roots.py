"""MCP `roots` capability — client tells the server which fs roots it can use.

Per the MCP spec (2025-11-25):
    - Client declares `capabilities.roots = {listChanged: bool}` at init.
    - Server requests `roots/list` to fetch the current list.
    - Each Root is `{uri: file://..., name: optional}`.

This module provides:
    - `Root`: lightweight dataclass mirroring the MCP type, decoupling
      callers from the MCP SDK import.
    - `path_to_root(path, name=...)`: build a Root from a `pathlib.Path`,
      automatically resolving to absolute and normalising to `file://`.
    - `RootsProvider`: a callable that returns `list[Root]` (sync) or
      `Awaitable[list[Root]]` (async). Wrapped into the MCP SDK's
      `list_roots_callback` shape by `MCPClient`.
    - `static_roots(roots)`: convenience constructor for an immutable
      provider (e.g. server-startup configuration).

Why a façade dataclass instead of re-using `mcp.types.Root` directly:
    - The `mcp` package is an optional dependency (only installed via the
      `mcp` extra group). Modules and tests that touch roots logic should
      not require the SDK to be importable.
    - Keeps the API independent of MCP SDK version churn (the SDK uses
      pydantic models; our boundary uses a stable dataclass).
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

RootsProvider = Callable[[], Union[list["Root"], Awaitable[list["Root"]]]]


@dataclass(slots=True, frozen=True)
class Root:
    """A filesystem root the client exposes to an MCP server.

    `uri` MUST start with `file://` (MCP spec). `name` is optional but
    recommended — operators see it in the server's logs and the LLM may
    use it as context. `meta` mirrors the spec's `_meta` slot for
    forward compatibility; defaults to None to keep mocks ergonomic.
    """

    uri: str
    name: str | None = None
    meta: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        # MCP spec is `file://` URIs only. We further require the local form
        # — `file:///path` (empty host, three slashes) or
        # `file://localhost/path` — and reject `file://otherhost/path`,
        # which would imply the server should reach across the network.
        # `pathlib.Path.as_uri()` always emits the three-slash form, so
        # path_to_root output passes; only hand-rolled URIs hit this branch.
        u = self.uri
        if not (u.startswith("file:///") or u.startswith("file://localhost/")):
            raise ValueError(
                f"Root uri must start with 'file:///' or 'file://localhost/' "
                f"(local file URI per MCP spec), got {u!r}"
            )


def path_to_root(path: str | Path, *, name: str | None = None) -> Root:
    """Build a `Root` from a filesystem path. The path is resolved to an
    absolute, normalised form (`Path.resolve()`) so relative paths in
    config files don't leak deployment-environment-specific roots to the
    server. `name` defaults to the directory's basename when omitted.
    """
    resolved = Path(path).expanduser().resolve()
    return Root(
        uri=resolved.as_uri(),
        name=name if name is not None else resolved.name,
    )


def static_roots(roots: list[Root]) -> RootsProvider:
    """Build a `RootsProvider` that always returns the same list (a copy
    each call so caller mutation can't poison subsequent server requests).

    Validates element types up-front (raises TypeError) so misuse like
    `static_roots(["file:///x"])` fails at the construction site instead
    of at the next `roots/list` request — early failures surface in the
    operator's startup logs, not three layers deep into MCP transport.
    """
    bad = [r for r in roots if not isinstance(r, Root)]
    if bad:
        raise TypeError(
            f"static_roots: every element must be a Root instance; "
            f"got {[type(r).__name__ for r in bad]}"
        )
    snapshot = tuple(roots)  # frozen storage

    def _provider() -> list[Root]:
        return list(snapshot)

    return _provider


async def call_roots_provider(provider: RootsProvider) -> list[Root]:
    """Invoke a sync/async provider and normalise to a concrete list.

    Engine-internal helper: `MCPClient` uses this in the `list_roots_callback`
    adapter so providers can be either sync (typical: static config) or
    async (e.g. dynamic per-tenant root resolution).
    """
    result = provider()
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, list):
        raise TypeError(
            f"RootsProvider must return list[Root], got {type(result).__name__}"
        )
    return result


__all__ = [
    "Root",
    "RootsProvider",
    "call_roots_provider",
    "path_to_root",
    "static_roots",
]
