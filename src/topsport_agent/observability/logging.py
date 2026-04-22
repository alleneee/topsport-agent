"""结构化 JSON 日志：为 stdlib logging 提供一层 JSONFormatter。

启用后每条日志输出为一行合法 JSON，便于 ELK/Loki/Datadog 等日志平台解析。
不引入 structlog/loguru 等新依赖，最小侵入。

用法：
    from topsport_agent.observability.logging import configure_json_logging
    configure_json_logging(level=logging.INFO)
    _logger.warning("session closed", extra={"session_id": sid, "tenant_id": t})

保留字段（永远在 payload 里）：ts / level / logger / msg。
extra={} 传入的任意字段会并入 payload（与保留字段同层）。
异常信息通过 exc 字段输出 traceback 字符串。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import IO, Any

_STANDARD_LOGRECORD_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})


class JSONFormatter(logging.Formatter):
    """把 LogRecord 序列化为单行 JSON。

    fields 顺序：ts, level, logger, msg, ...extra, exc?
    非 JSON 原生类型回退到 str() 表示，保证 dumps 不抛。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOGRECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_json_logging(
    *,
    level: int = logging.INFO,
    stream: IO[str] | None = None,
    root_logger: logging.Logger | None = None,
) -> logging.Handler:
    """在 root logger 上挂一个 stderr/stdout 的 JSON handler（幂等）。

    重复调用：先移除标记为 `_topsport_json_handler=True` 的既有 handler，避免重复。
    返回新装的 handler，便于测试或自定义 handler 级别。
    """
    target = root_logger if root_logger is not None else logging.getLogger()
    for existing in list(target.handlers):
        if getattr(existing, "_topsport_json_handler", False):
            target.removeHandler(existing)
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(JSONFormatter())
    handler.setLevel(level)
    setattr(handler, "_topsport_json_handler", True)
    target.addHandler(handler)
    if target.level == logging.NOTSET or target.level > level:
        target.setLevel(level)
    return handler


__all__ = ["JSONFormatter", "configure_json_logging"]
