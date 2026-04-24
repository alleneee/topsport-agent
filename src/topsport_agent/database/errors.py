"""Stable error hierarchy. Backends translate driver-native exceptions to these."""

from __future__ import annotations


class DatabaseError(Exception):
    """Base class for any database-layer failure."""


class ConnectionError(DatabaseError):  # noqa: A001 — shadowing builtin is intentional and namespaced
    """Pool or connection setup/teardown failed."""


class QueryError(DatabaseError):
    """SQL syntax / execution failed."""


class IntegrityError(QueryError):
    """Unique / foreign-key / check constraint violation."""


class TransactionError(DatabaseError):
    """Transaction begin / commit / rollback failed."""


__all__ = [
    "ConnectionError",
    "DatabaseError",
    "IntegrityError",
    "QueryError",
    "TransactionError",
]
