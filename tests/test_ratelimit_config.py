import dataclasses

import pytest

from topsport_agent.ratelimit.config import RateLimitConfig


def test_defaults_disabled() -> None:
    cfg = RateLimitConfig()
    assert cfg.enabled is False
    assert cfg.redis_url is None
    assert cfg.window_seconds == 60
    assert cfg.per_ip_limit == 300
    assert cfg.per_principal_limit == 60
    assert cfg.per_tenant_limit == 1000
    assert cfg.per_route_default == 0
    assert cfg.per_route_limits == {}
    assert "/health" in cfg.exempt_paths
    assert "/metrics" in cfg.exempt_paths
    assert "/docs" in cfg.exempt_paths
    assert "/openapi.json" in cfg.exempt_paths
    assert cfg.trust_forwarded_for is False
    assert cfg.fail_open_on_redis_error is True


def test_config_is_frozen() -> None:
    cfg = RateLimitConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.enabled = True  # type: ignore[misc]


def test_custom_config() -> None:
    cfg = RateLimitConfig(
        enabled=True,
        redis_url="redis://localhost:6379/0",
        per_ip_limit=500,
        per_route_limits={"/chat": 20},
    )
    assert cfg.enabled is True
    assert cfg.per_route_limits == {"/chat": 20}
