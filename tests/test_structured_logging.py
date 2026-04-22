"""测试 observability.logging 的 JSONFormatter 与 configure_json_logging。

覆盖：
- 每条输出是合法 JSON，含保留字段（ts/level/logger/msg）
- extra={} 字段并入 payload
- 异常通过 exc 字段输出
- 幂等：重复 configure 不重复挂 handler
- 非 JSON 原生类型走 default=str 不抛
- ts 是 ISO-8601 UTC
"""

from __future__ import annotations

import io
import json
import logging
import re

import pytest

from topsport_agent.observability.logging import (
    JSONFormatter,
    configure_json_logging,
)


@pytest.fixture
def isolated_logger():
    """隔离测试 logger，避免污染 root logger。"""
    logger = logging.getLogger("topsport_agent.test.isolated")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield logger
    for h in list(logger.handlers):
        logger.removeHandler(h)


def _json_lines(buf: io.StringIO) -> list[dict]:
    raw = buf.getvalue()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def test_json_formatter_outputs_reserved_fields(isolated_logger):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JSONFormatter())
    isolated_logger.addHandler(handler)

    isolated_logger.info("hello world")

    records = _json_lines(buf)
    assert len(records) == 1
    r = records[0]
    assert r["level"] == "INFO"
    assert r["logger"] == "topsport_agent.test.isolated"
    assert r["msg"] == "hello world"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T.*\+00:00$", r["ts"])


def test_extra_fields_merge_into_payload(isolated_logger):
    buf = io.StringIO()
    isolated_logger.addHandler(
        _handler(buf)
    )

    isolated_logger.warning(
        "session closed",
        extra={"session_id": "sess-abc", "tenant_id": "t1", "principal": "niko"},
    )

    r = _json_lines(buf)[0]
    assert r["session_id"] == "sess-abc"
    assert r["tenant_id"] == "t1"
    assert r["principal"] == "niko"
    assert r["msg"] == "session closed"


def test_exception_renders_as_exc_field(isolated_logger):
    buf = io.StringIO()
    isolated_logger.addHandler(_handler(buf))

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        isolated_logger.exception("something failed")

    r = _json_lines(buf)[0]
    assert r["level"] == "ERROR"
    assert "exc" in r
    assert "RuntimeError" in r["exc"]
    assert "boom" in r["exc"]


def test_non_json_native_extra_falls_back_to_str(isolated_logger):
    buf = io.StringIO()
    isolated_logger.addHandler(_handler(buf))

    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    isolated_logger.info("with opaque", extra={"obj": Opaque()})

    r = _json_lines(buf)[0]
    assert r["obj"] == "<Opaque>"


def test_configure_is_idempotent():
    # 在一个我们拥有的 logger 上测试，不污染 root
    logger = logging.getLogger("topsport_agent.test.configure_idempotent")
    logger.handlers.clear()

    buf1 = io.StringIO()
    configure_json_logging(level=logging.INFO, stream=buf1, root_logger=logger)
    buf2 = io.StringIO()
    configure_json_logging(level=logging.INFO, stream=buf2, root_logger=logger)

    json_handlers = [
        h for h in logger.handlers if getattr(h, "_topsport_json_handler", False)
    ]
    assert len(json_handlers) == 1


def test_configure_writes_real_json(isolated_logger):
    buf = io.StringIO()
    configure_json_logging(level=logging.INFO, stream=buf, root_logger=isolated_logger)
    isolated_logger.info("through configure")

    r = _json_lines(buf)[0]
    assert r["msg"] == "through configure"
    assert r["level"] == "INFO"


def test_standard_logrecord_fields_not_leaked(isolated_logger):
    """LogRecord 的内部字段（pathname/funcName 等）不应作为 payload 字段外泄。"""
    buf = io.StringIO()
    isolated_logger.addHandler(_handler(buf))

    isolated_logger.info("plain")

    r = _json_lines(buf)[0]
    for leak in ("pathname", "filename", "funcName", "lineno", "module"):
        assert leak not in r, f"{leak} should not leak into JSON payload"


def _handler(buf: io.StringIO) -> logging.Handler:
    h = logging.StreamHandler(buf)
    h.setFormatter(JSONFormatter())
    return h
