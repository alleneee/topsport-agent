import pytest

from topsport_agent.database.backends.null import NullGateway
from topsport_agent.database.gateway import Database


def test_null_gateway_satisfies_database_protocol() -> None:
    gw = NullGateway()
    assert isinstance(gw, Database)  # runtime_checkable Protocol


def test_dialect_is_null() -> None:
    assert NullGateway().dialect == "null"


@pytest.mark.asyncio
async def test_lifecycle_is_noop() -> None:
    gw = NullGateway()
    await gw.connect()
    await gw.connect()  # idempotent
    assert await gw.health_check() is True
    await gw.close()
    await gw.close()  # idempotent


@pytest.mark.asyncio
async def test_queries_raise_disabled_error() -> None:
    gw = NullGateway()
    with pytest.raises(RuntimeError, match="database disabled"):
        await gw.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="database disabled"):
        await gw.fetch_one("SELECT 1")
    with pytest.raises(RuntimeError, match="database disabled"):
        await gw.fetch_all("SELECT 1")
    with pytest.raises(RuntimeError, match="database disabled"):
        await gw.fetch_val("SELECT 1")


@pytest.mark.asyncio
async def test_transaction_raises_disabled_error() -> None:
    gw = NullGateway()
    with pytest.raises(RuntimeError, match="database disabled"):
        async with gw.transaction():
            pass
