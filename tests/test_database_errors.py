import pytest

from topsport_agent.database.errors import (
    ConnectionError as DBConnectionError,
    DatabaseError,
    IntegrityError,
    QueryError,
    TransactionError,
)


def test_all_errors_inherit_database_error() -> None:
    assert issubclass(DBConnectionError, DatabaseError)
    assert issubclass(QueryError, DatabaseError)
    assert issubclass(TransactionError, DatabaseError)
    assert issubclass(IntegrityError, QueryError)  # IntegrityError is a QueryError


def test_database_error_is_plain_exception() -> None:
    # base class is plain Exception — callers can catch broadly if needed
    assert issubclass(DatabaseError, Exception)


def test_errors_carry_messages() -> None:
    with pytest.raises(DBConnectionError, match="pool failed"):
        raise DBConnectionError("pool failed")
