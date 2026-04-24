import pytest

from topsport_agent.database.backends.mysql import MySQLGateway
from topsport_agent.database.backends.sqlite import SqliteGateway
from topsport_agent.database.config import DatabaseConfig


def test_mysql_gateway_raises_on_construction() -> None:
    with pytest.raises(NotImplementedError, match="mysql backend not implemented"):
        MySQLGateway(DatabaseConfig(backend="mysql", url="mysql://x"))


def test_sqlite_gateway_raises_on_construction() -> None:
    with pytest.raises(NotImplementedError, match="sqlite backend not implemented"):
        SqliteGateway(DatabaseConfig(backend="sqlite", url="sqlite:///x"))
