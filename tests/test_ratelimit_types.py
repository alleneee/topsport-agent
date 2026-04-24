import dataclasses

import pytest

from topsport_agent.ratelimit.types import (
    RateLimitDecision,
    RateLimitRule,
    RateLimitScope,
)


def test_scope_values() -> None:
    assert RateLimitScope.IP == "ip"
    assert RateLimitScope.PRINCIPAL == "principal"
    assert RateLimitScope.TENANT == "tenant"
    assert RateLimitScope.ROUTE == "route"


def test_rule_is_frozen() -> None:
    rule = RateLimitRule(
        scope=RateLimitScope.IP,
        identity="1.2.3.4",
        limit=100,
        window_seconds=60,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.limit = 200  # type: ignore[misc]


def test_decision_is_frozen() -> None:
    decision = RateLimitDecision(
        allowed=True,
        denied_scope=None,
        limit=100,
        remaining=99,
        reset_at_ms=123456,
        retry_after_seconds=0,
    )
    assert decision.allowed is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.allowed = False  # type: ignore[misc]


def test_denied_decision_has_scope() -> None:
    decision = RateLimitDecision(
        allowed=False,
        denied_scope=RateLimitScope.TENANT,
        limit=1000,
        remaining=0,
        reset_at_ms=123456,
        retry_after_seconds=30,
    )
    assert decision.allowed is False
    assert decision.denied_scope == RateLimitScope.TENANT
    assert decision.retry_after_seconds == 30
