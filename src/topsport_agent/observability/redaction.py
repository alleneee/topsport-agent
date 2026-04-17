"""观测数据脱敏：LangfuseTracer 上报前先过滤 payload，屏蔽秘钥/凭据/PII。

SimpleRedactor 支持按字段名匹配和按值的正则匹配两种规则，返回深拷贝以免污染
事件原始 payload。用户可注入自定义 Callable[[dict], dict] 作为更灵活的策略。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

_REDACTED = "[REDACTED]"

# 字段名（key）命中则整个 value 替换为 [REDACTED]。大小写不敏感，子串匹配。
_DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "secret_key",
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "bearer",
        "cookie",
        "set-cookie",
        "x-api-key",
    }
)

# 值级正则：典型凭据格式，命中则替换整个字符串（保守替换）。
_DEFAULT_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),         # OpenAI 风格
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),     # Anthropic
    re.compile(r"AKIA[0-9A-Z]{16}"),                # AWS access key id
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}=*"),
)

Redactor = Callable[[Any], Any]


@dataclass(slots=True, frozen=True)
class SimpleRedactor:
    """字段名匹配 + 值正则匹配的脱敏器。"""

    sensitive_keys: frozenset[str] = field(default_factory=lambda: _DEFAULT_SENSITIVE_KEYS)
    value_patterns: tuple[re.Pattern[str], ...] = field(default_factory=lambda: _DEFAULT_VALUE_PATTERNS)

    def __call__(self, payload: Any) -> Any:
        return self._walk(payload)

    def _walk(self, value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[Any, Any] = {}
            for k, v in value.items():
                if isinstance(k, str) and self._key_sensitive(k):
                    out[k] = _REDACTED
                else:
                    out[k] = self._walk(v)
            return out
        if isinstance(value, list):
            return [self._walk(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._walk(v) for v in value)
        if isinstance(value, str):
            return self._scrub_str(value)
        return value

    def _key_sensitive(self, key: str) -> bool:
        low = key.lower()
        for kw in self.sensitive_keys:
            if kw in low:
                return True
        return False

    def _scrub_str(self, s: str) -> str:
        for pat in self.value_patterns:
            if pat.search(s):
                return _REDACTED
        return s


def default_redactor() -> SimpleRedactor:
    return SimpleRedactor()


def validate_base_url(base_url: str, allowlist: Iterable[str]) -> None:
    """base_url 必须 startswith allowlist 中某一条；allowlist 为空时跳过检查。"""
    allowed = [a for a in allowlist if a]
    if not allowed:
        return
    for prefix in allowed:
        if base_url.startswith(prefix):
            return
    raise ValueError(
        f"langfuse base_url {base_url!r} is not in allowlist; "
        f"permitted prefixes: {allowed}"
    )
