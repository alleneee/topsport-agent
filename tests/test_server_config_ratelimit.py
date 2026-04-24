import pytest

from topsport_agent.server.config import ServerConfig


def test_ratelimit_defaults_disabled() -> None:
    cfg = ServerConfig()
    assert cfg.enable_rate_limit is False
    assert cfg.ratelimit_redis_url is None
    assert cfg.ratelimit_window_seconds == 60
    assert cfg.ratelimit_per_ip == 300
    assert cfg.ratelimit_per_principal == 60
    assert cfg.ratelimit_per_tenant == 1000
    assert cfg.ratelimit_per_route_default == 0
    assert cfg.ratelimit_routes == {}
    assert cfg.ratelimit_trust_forwarded_for is False
    assert cfg.ratelimit_fail_open is True


def test_ratelimit_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_RATE_LIMIT", "true")
    monkeypatch.setenv("RATELIMIT_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RATELIMIT_PER_IP", "500")
    monkeypatch.setenv("RATELIMIT_PER_TENANT", "2000")
    monkeypatch.setenv("RATELIMIT_WINDOW_SECONDS", "30")
    monkeypatch.setenv("RATELIMIT_ROUTES", '{"/chat": 20}')
    monkeypatch.setenv("RATELIMIT_TRUST_FORWARDED_FOR", "true")
    monkeypatch.setenv("RATELIMIT_FAIL_OPEN", "false")
    monkeypatch.setenv("API_KEY", "dummy")

    cfg = ServerConfig.from_env()
    assert cfg.enable_rate_limit is True
    assert cfg.ratelimit_redis_url == "redis://localhost:6379/0"
    assert cfg.ratelimit_per_ip == 500
    assert cfg.ratelimit_per_tenant == 2000
    assert cfg.ratelimit_window_seconds == 30
    assert cfg.ratelimit_routes == {"/chat": 20}
    assert cfg.ratelimit_trust_forwarded_for is True
    assert cfg.ratelimit_fail_open is False


def test_ratelimit_routes_empty_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RATELIMIT_ROUTES", raising=False)
    monkeypatch.setenv("API_KEY", "dummy")

    cfg = ServerConfig.from_env()
    assert cfg.ratelimit_routes == {}


def test_ratelimit_routes_invalid_json_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RATELIMIT_ROUTES", "not-json")
    monkeypatch.setenv("API_KEY", "dummy")

    with pytest.raises(ValueError):
        ServerConfig.from_env()
