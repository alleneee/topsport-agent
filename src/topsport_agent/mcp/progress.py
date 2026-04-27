"""MCP `progress` notification consumption.

Per the MCP spec (2025-11-25):
    - Client sends a request with `_meta.progressToken`.
    - Server pushes `notifications/progress` with
      `{progressToken, progress, total?, message?}` until completion.

The MCP Python SDK exposes this as a per-call argument:
    `ClientSession.call_tool(..., progress_callback: ProgressFnT)`
where `ProgressFnT = Callable[[progress: float, total: float | None,
message: str | None], Awaitable[None] | None]`.

This module provides:
    - `ProgressCallback`: SDK-shaped Callable type re-exported so callers
      don't import the SDK directly (the `mcp` extra is optional).
    - `default_progress_callback(client_name)`: factory routing progress
      events into `topsport_agent.mcp.progress.<client_name>` Python
      logger. Operators get a free observability signal for long-running
      MCP tool calls without subscribing manually.
    - `wrap_progress_callback(callback)`: exception-isolation wrapper.
      MCP server emits progress notifications on its own schedule; a
      buggy callback should not abort the in-flight tool call. Mirrors
      the safe-callback pattern used by 5.3 logging.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Union

# Matches mcp.shared.session.ProgressFnT shape (sync OR async). SDK
# auto-awaits the return value if it's an Awaitable.
ProgressCallback = Callable[
    [float, float | None, str | None],
    Union[Awaitable[None], None],
]


def default_progress_callback(
    client_name: str,
    *,
    level: int = logging.INFO,
    sample_every: int | None = None,
) -> ProgressCallback:
    """Build a progress callback that emits log records to
    `topsport_agent.mcp.progress.<client_name>`.

    `level`: stdlib logging level (defaults INFO). DEBUG is the typical
    pick once an operator notices the progress stream is loud and they
    only want WARNING+ to surface in the production tail.
    `sample_every`: when set to N>0, only every Nth update is logged
    (counter is per-callback instance). None / 0 → no sampling. Useful
    for servers that emit progress at 100ms intervals — sample_every=10
    cuts log volume by 90% while still showing motion.

    Format: "progress=<n>/<total> message=<msg>" with `total` and
    `message` omitted when None — keeps the line greppable. Long-running
    tool calls show up in the same pipeline as ordinary MCP server logs
    (5.3) so ops can build a single dashboard for "MCP activity".

    Note: `client_name` is appended to the logger as a final dotted
    component, so `client_name="a.b"` produces logger
    `topsport_agent.mcp.progress.a.b` — events propagate up the
    `topsport_agent.mcp.progress.a` parent logger via stdlib hierarchy.
    Use only `[A-Za-z0-9_-]` characters in client names if you want a
    flat tree.
    """
    logger_name = f"topsport_agent.mcp.progress.{client_name}"
    counter = {"n": 0}

    def _callback(progress: float, total: float | None, message: str | None) -> None:
        if sample_every and sample_every > 0:
            counter["n"] += 1
            if counter["n"] % sample_every != 1:  # 1, N+1, 2N+1...
                return
        log = logging.getLogger(logger_name)
        if total is not None and message is not None:
            log.log(level, "progress=%s/%s message=%s", progress, total, message)
        elif total is not None:
            log.log(level, "progress=%s/%s", progress, total)
        elif message is not None:
            log.log(level, "progress=%s message=%s", progress, message)
        else:
            log.log(level, "progress=%s", progress)

    return _callback


def wrap_progress_callback(
    callback: ProgressCallback,
    *,
    client_name: str,
) -> ProgressCallback:
    """Wrap `callback` so exceptions inside don't propagate to the SDK
    notification dispatch loop or — worse — abort the tool call that's
    awaiting completion.

    `client_name` is required (keyword-only, no default) so when an
    isolated exception is logged the operator can immediately see which
    client failed — a `?` placeholder default would defeat the
    observability gain.

    Sync and async callbacks both supported; awaitable returns are
    awaited inside the wrapper before the SDK proceeds.
    """
    _logger = logging.getLogger(__name__)

    async def _safe(progress: float, total: float | None, message: str | None) -> None:
        try:
            result = callback(progress, total, message)
            if inspect.isawaitable(result):
                await result
        except Exception:
            _logger.warning(
                "mcp client %r progress_callback raised; dropping update "
                "(progress=%s total=%s message=%s)",
                client_name, progress, total, message, exc_info=True,
            )

    return _safe


__all__ = [
    "ProgressCallback",
    "default_progress_callback",
    "wrap_progress_callback",
]
