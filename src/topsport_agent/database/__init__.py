"""Database Gateway abstraction — pluggable multi-backend skeleton."""

from .backends.null import NullGateway
from .config import DatabaseConfig
from .errors import (
    ConnectionError,
    DatabaseError,
    IntegrityError,
    QueryError,
    TransactionError,
)
from .factory import create_database
from .gateway import Database, Transaction

__all__ = [
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
]
