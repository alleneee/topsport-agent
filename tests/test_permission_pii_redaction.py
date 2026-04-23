from __future__ import annotations

import re

from topsport_agent.engine.permission.redaction import (
    PIIRedactor,
    RedactionPattern,
)


def test_redactor_default_patterns_mask_email():
    r = PIIRedactor.with_defaults()
    out = r.redact_and_truncate({"text": "Contact alice@corp.com please"})
    assert "alice@corp.com" not in out["text"]
    assert "[email]" in out["text"]


def test_redactor_masks_openai_style_token():
    r = PIIRedactor.with_defaults()
    out = r.redact_and_truncate(
        {"auth": "sk-abcdefghij1234567890abcdefghij"}
    )
    assert "sk-abcdefghij" not in out["auth"]
    assert "[token]" in out["auth"]


def test_redactor_walks_nested_dicts_and_lists():
    r = PIIRedactor.with_defaults()
    out = r.redact_and_truncate({
        "user": {"email": "bob@x.com"},
        "tokens": ["sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
    })
    assert "bob@x.com" not in str(out)
    assert "[email]" in str(out)


def test_redactor_truncates_large_payload_to_4kb():
    r = PIIRedactor.with_defaults()
    big = "X" * 10_000
    out = r.redact_and_truncate({"data": big})
    import json
    serialized = json.dumps(out)
    assert len(serialized) <= 4200  # 4 KB + small sentinel overhead
    assert out.get("__truncated__") is True


def test_redactor_small_payload_not_truncated():
    r = PIIRedactor.with_defaults()
    out = r.redact_and_truncate({"data": "tiny"})
    assert "__truncated__" not in out


def test_redactor_custom_pattern():
    r = PIIRedactor([
        RedactionPattern(
            pattern=re.compile(r"EMP-\d{6}"),
            replacement="[emp-id]",
        ),
    ])
    out = r.redact_and_truncate({"note": "user EMP-123456 logged in"})
    assert "EMP-123456" not in out["note"]
    assert "[emp-id]" in out["note"]


def test_redactor_idempotent_on_non_string_values():
    r = PIIRedactor.with_defaults()
    out = r.redact_and_truncate({"n": 42, "b": True, "n2": None})
    assert out["n"] == 42
    assert out["b"] is True
    assert out["n2"] is None
