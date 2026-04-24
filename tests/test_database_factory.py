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
