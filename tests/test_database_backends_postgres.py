import importlib
import importlib.util

import pytest

from topsport_agent.database.config import DatabaseConfig
from topsport_agent.database.errors import ConnectionError as DBConnectionError

_asyncpg_available = importlib.util.find_spec("asyncpg") is not None
requires_asyncpg = pytest.mark.skipif(
    not _asyncpg_available, reason="asyncpg not installed (uv sync --group db)"
)


@requires_asyncpg
def test_postgres_gateway_dialect() -> None:
    from topsport_agent.database.backends.postgres import PostgresGateway

    gw = PostgresGateway(DatabaseConfig(backend="postgres", url="postgresql://x"))
    assert gw.dialect == "postgres"


@requires_asyncpg
@pytest.mark.asyncio
async def test_postgres_gateway_connect_failure_wraps_as_db_connection_error() -> None:
    """Unreachable host → our ConnectionError, not raw asyncpg exception."""
    from topsport_agent.database.backends.postgres import PostgresGateway

    gw = PostgresGateway(
        DatabaseConfig(
            backend="postgres",
            url="postgresql://nonexistent.invalid:5432/db",
            timeout_seconds=1.0,
        )
    )
    with pytest.raises(DBConnectionError):
        await gw.connect()


@requires_asyncpg
@pytest.mark.asyncio
async def test_postgres_gateway_queries_raise_not_implemented() -> None:
    from topsport_agent.database.backends.postgres import PostgresGateway

    gw = PostgresGateway(DatabaseConfig(backend="postgres", url="postgresql://x"))
    with pytest.raises(NotImplementedError):
        await gw.execute("SELECT 1")
    with pytest.raises(NotImplementedError):
        await gw.fetch_one("SELECT 1")
    with pytest.raises(NotImplementedError):
        await gw.fetch_all("SELECT 1")
    with pytest.raises(NotImplementedError):
        await gw.fetch_val("SELECT 1")
    with pytest.raises(NotImplementedError):
        async with gw.transaction():
            pass


@requires_asyncpg
@pytest.mark.asyncio
async def test_postgres_gateway_close_is_idempotent_before_connect() -> None:
    """close() before connect() must not raise; required for error paths in lifespan."""
    from topsport_agent.database.backends.postgres import PostgresGateway

    gw = PostgresGateway(DatabaseConfig(backend="postgres", url="postgresql://x"))
    await gw.close()
    await gw.close()


def test_postgres_gateway_module_importable() -> None:
    """The module imports even without asyncpg installed."""
    mod = importlib.import_module("topsport_agent.database.backends.postgres")
    assert hasattr(mod, "PostgresGateway")
