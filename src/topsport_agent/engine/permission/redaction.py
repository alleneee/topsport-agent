"""PII redactor for audit_preview payloads.

Applied before writing `AuditEntry.args_preview`. Walks nested dict/list/tuple
structures, applies regex substitutions to every string leaf, then serializes
the result. If serialized JSON exceeds `max_bytes`, the payload is replaced
by a truncated variant with a `__truncated__: True` sentinel.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["PIIRedactor", "RedactionPattern"]

_DEFAULT_MAX_BYTES = 4096


@dataclass(frozen=True, slots=True)
class RedactionPattern:
    pattern: re.Pattern[str]
    replacement: str


# First-match-wins patterns for the common case.
_DEFAULT_PATTERNS: tuple[RedactionPattern, ...] = (
    # OpenAI / Anthropic style tokens must match before generic word boundary rules
    RedactionPattern(
        pattern=re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        replacement="[token]",
    ),
    RedactionPattern(
        pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
        replacement="[aws-key]",
    ),
    RedactionPattern(
        pattern=re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
        replacement="[email]",
    ),
    # Luhn-plausible 13-19 digit runs
    RedactionPattern(
        pattern=re.compile(r"\b\d{13,19}\b"),
        replacement="[cc]",
    ),
    # Internationalized phone (loose — catches too eagerly; tighten via custom if needed)
    RedactionPattern(
        pattern=re.compile(r"\+?\d[\d\s\-().]{8,}\d"),
        replacement="[phone]",
    ),
)


class PIIRedactor:
    def __init__(
        self,
        patterns: Sequence[RedactionPattern],
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self._patterns = tuple(patterns)
        self._max_bytes = max_bytes

    @classmethod
    def with_defaults(cls, max_bytes: int = _DEFAULT_MAX_BYTES) -> "PIIRedactor":
        return cls(_DEFAULT_PATTERNS, max_bytes=max_bytes)

    def redact_and_truncate(self, payload: dict[str, Any]) -> dict[str, Any]:
        redacted = self._walk(payload)
        serialized = json.dumps(redacted, ensure_ascii=False, default=str)
        if len(serialized.encode("utf-8")) <= self._max_bytes:
            return redacted
        # Truncate: keep a prefix of the serialized form inside a sentinel dict.
        truncated_json = serialized[: self._max_bytes]
        return {
            "__truncated__": True,
            "__preview__": truncated_json,
            "__original_size__": len(serialized),
        }

    def _walk(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._apply(value)
        if isinstance(value, dict):
            return {k: self._walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._walk(v) for v in value]
        return value

    def _apply(self, text: str) -> str:
        for pat in self._patterns:
            text = pat.pattern.sub(pat.replacement, text)
        return text
