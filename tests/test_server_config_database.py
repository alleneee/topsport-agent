import pytest

from topsport_agent.server.config import ServerConfig


def test_defaults_disable_database() -> None:
    cfg = ServerConfig()
    assert cfg.enable_database is False
    assert cfg.database_backend == "postgres"  # default when enabled
    assert cfg.database_url is None
    assert cfg.database_pool_min == 1
    assert cfg.database_pool_max == 10
    assert cfg.database_timeout_seconds == 30.0


def test_env_parsing_enable_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_DATABASE", "true")
    monkeypatch.setenv("DATABASE_BACKEND", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("DATABASE_POOL_MIN", "2")
    monkeypatch.setenv("DATABASE_POOL_MAX", "5")
    monkeypatch.setenv("API_KEY", "dummy")

    cfg = ServerConfig.from_env()
    assert cfg.enable_database is True
    assert cfg.database_backend == "postgres"
    assert cfg.database_url == "postgresql://localhost/x"
    assert cfg.database_pool_min == 2
    assert cfg.database_pool_max == 5


def test_env_parsing_database_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for k in (
        "ENABLE_DATABASE",
        "DATABASE_BACKEND",
        "DATABASE_URL",
        "DATABASE_POOL_MIN",
        "DATABASE_POOL_MAX",
        "DATABASE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("API_KEY", "dummy")

    cfg = ServerConfig.from_env()
    assert cfg.enable_database is False
