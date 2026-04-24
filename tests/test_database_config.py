import dataclasses

import pytest

from topsport_agent.database.config import DatabaseConfig


def test_default_config_is_null_backend() -> None:
    cfg = DatabaseConfig()
    assert cfg.backend == "null"
    assert cfg.url is None
    assert cfg.pool_min == 1
    assert cfg.pool_max == 10
    assert cfg.timeout_seconds == 30.0


def test_config_is_frozen() -> None:
    cfg = DatabaseConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.backend = "postgres"  # type: ignore[misc]


def test_custom_config_round_trip() -> None:
    cfg = DatabaseConfig(
        backend="postgres",
        url="postgresql://localhost/test",
        pool_min=2,
        pool_max=20,
        timeout_seconds=5.0,
    )
    assert cfg.backend == "postgres"
    assert cfg.url == "postgresql://localhost/test"
    assert cfg.pool_min == 2
    assert cfg.pool_max == 20
    assert cfg.timeout_seconds == 5.0
