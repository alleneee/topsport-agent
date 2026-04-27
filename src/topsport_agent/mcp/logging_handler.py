"""MCP `logging` capability — consume server log notifications.

Per the MCP spec (2025-11-25):
    - Server declares `capabilities.logging = {}` at init.
    - Server pushes `notifications/message` events with
      `{level, logger, data, meta}`.
    - Client may call `logging/setLevel` to adjust the server's
      threshold (defaults vary per server; some send all DEBUG).

This module provides:
    - `MCPLogLevel`: the 8-tier MCP severity ladder, exported as
      string constants so callers can avoid pulling pydantic types
      from the optional `mcp` SDK.
    - `mcp_level_to_python(level)`: maps MCP -> stdlib `logging` levels
      so server messages flow into the agent's existing structured
      logging pipeline (ELK / Loki / FastAPI debug console / pytest
      caplog) without bespoke handlers.
    - `default_logging_callback(client_name)`: factory that returns a
      ListChangedCallback-shaped handler logging into
      `topsport_agent.mcp.server.<client_name>` so operators can
      filter / route per-server in their logging config.

Why a stable string ladder instead of importing `mcp.types.LoggingLevel`
directly: the `mcp` SDK is optional. Anyone wanting to *configure* the
threshold (operators, ServerConfig parsing, tests) shouldn't need the
SDK installed; only `MCPClient`'s callback path imports it lazily.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

# 8-tier MCP severity ladder (RFC 5424 syslog form).
MCPLogLevel = Literal[
    "debug", "info", "notice", "warning", "error", "critical", "alert", "emergency",
]

LoggingCallback = Callable[[Any], Awaitable[None]]
"""Callable signature: `(LoggingMessageNotificationParams) -> Awaitable[None]`.

Type erased to `Any` so callers don't transitively need the MCP SDK
import; `MCPClient` adapts to the SDK's `LoggingFnT` shape internally.
"""


# stdlib logging has no dedicated NOTICE/ALERT/EMERGENCY levels. We pick
# pragmatic mappings that preserve total ordering and let stdlib filters
# keep working (NOTICE just above INFO, ALERT == EMERGENCY == CRITICAL).
_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO + 5,  # 25 — between INFO (20) and WARNING (30)
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "alert": logging.CRITICAL,
    "emergency": logging.CRITICAL,
}


def mcp_level_to_python(level: str) -> int:
    """Map an MCP log level string to a stdlib `logging` integer level.

    Unknown levels fall back to INFO (with a one-time WARNING) — keeps
    misbehaving servers visible without crashing the dispatch loop.
    """
    mapped = _LEVEL_MAP.get(level.lower() if level else "")
    if mapped is None:
        logging.getLogger(__name__).warning(
            "unknown MCP log level %r; falling back to INFO", level,
        )
        return logging.INFO
    return mapped


def default_logging_callback(client_name: str) -> LoggingCallback:
    """Build a logging callback that routes server messages into the
    `topsport_agent.mcp.server.<client_name>` Python logger.

    `data` field is dispatched verbatim: strings → message, dicts → as
    extra. `logger` field (the server's own logger name) is appended to
    the Python logger name as `.<server-logger>` so operators can grep
    by both client and server module.
    """
    # NOTE: client_name 中的破折号（如 "brave-search"）在 stdlib `logging.config.
    # dictConfig` 下没问题，但 `fileConfig` 的 INI section 名禁止破折号；用
    # fileConfig 的 ops 应在 logger 配置里 quote 名称或额外做 normalisation。
    base_logger_name = f"topsport_agent.mcp.server.{client_name}"

    async def _callback(params: Any) -> None:
        level_str = getattr(params, "level", "info")
        server_logger_name = getattr(params, "logger", None)
        data = getattr(params, "data", None)
        meta = getattr(params, "meta", None)

        target_name = (
            f"{base_logger_name}.{server_logger_name}"
            if server_logger_name else base_logger_name
        )
        logger = logging.getLogger(target_name)
        py_level = mcp_level_to_python(level_str)

        extra: dict[str, Any] = {"mcp_level": level_str}
        if meta is not None:
            # spec: meta 用于 trace context（trace_id/span_id 等）。落 ELK 时
            # 保留它便于跨系统关联（Langfuse / OTEL）。
            extra["mcp_meta"] = meta

        # Dict data: serialise to a stable "k=v" message for grep + ship the
        # full dict via `extra` so structured-logging handlers (json formatter)
        # can index per-field. Reserved LogRecord names (`name` / `msg` /
        # `args` etc.) cannot be used as extra keys — we only set
        # `mcp_data` / `mcp_level` / `mcp_meta`, all safe.
        if isinstance(data, dict):
            try:
                summary = " ".join(f"{k}={v!r}" for k, v in data.items())
            except Exception:
                summary = str(data)
            extra["mcp_data"] = data
            logger.log(py_level, "mcp.message %s", summary, extra=extra)
        else:
            logger.log(
                py_level, "%s",
                "" if data is None else str(data),
                extra=extra,
            )

    return _callback


__all__ = [
    "LoggingCallback",
    "MCPLogLevel",
    "default_logging_callback",
    "mcp_level_to_python",
]
