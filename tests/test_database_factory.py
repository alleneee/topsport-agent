import importlib
import importlib.util

import pytest

from topsport_agent.database.backends.null import NullGateway
from topsport_agent.database.config import DatabaseConfig
from topsport_agent.database.factory import create_database

_asyncpg_available = importlib.util.find_spec("asyncpg") is not None


def test_factory_null_backend_returns_null_gateway() -> None:
    db = create_database(DatabaseConfig(backend="null"))
    assert isinstance(db, NullGateway)


def test_factory_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        create_database(DatabaseConfig(backend="mongodb"))


def test_factory_mysql_placeholder_raises() -> None:
    with pytest.raises(NotImplementedError, match="mysql backend not implemented"):
        create_database(DatabaseConfig(backend="mysql"))


def test_factory_sqlite_placeholder_raises() -> None:
    with pytest.raises(NotImplementedError, match="sqlite backend not implemented"):
        create_database(DatabaseConfig(backend="sqlite"))


@pytest.mark.skipif(not _asyncpg_available, reason="asyncpg not installed")
def test_factory_postgres_returns_postgres_gateway() -> None:
    from topsport_agent.database.backends.postgres import PostgresGateway

    db = create_database(
        DatabaseConfig(backend="postgres", url="postgresql://x")
    )
    assert isinstance(db, PostgresGateway)


def test_factory_postgres_missing_asyncpg_gives_friendly_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When asyncpg is unavailable, the factory must wrap the ImportError with
    a install-hint message. Regression test for the bug where the wrapping was
    around `import_module('...postgres')` (which always succeeds because
    postgres.py has no top-level asyncpg import) instead of around the
    `PostgresGateway(config)` construction call (which is where asyncpg is
    actually imported via importlib).
    """
    import sys

    # Force any `importlib.import_module("asyncpg")` call inside
    # PostgresGateway.__init__ to raise ImportError.
    monkeypatch.setitem(sys.modules, "asyncpg", None)

    with pytest.raises(ImportError, match="uv sync --group db"):
        create_database(
            DatabaseConfig(backend="postgres", url="postgresql://x")
        )


def test_public_package_exports() -> None:
    import topsport_agent.database as db_pkg

    expected = {
        "ConnectionError",
        "Database",
        "DatabaseConfig",
        "DatabaseError",
        "IntegrityError",
        "NullGateway",
        "QueryError",
        "Transaction",
        "TransactionError",
        "create_database",
    }
    for name in expected:
        assert hasattr(db_pkg, name), f"missing public export: {name}"
    assert set(db_pkg.__all__) == expected
