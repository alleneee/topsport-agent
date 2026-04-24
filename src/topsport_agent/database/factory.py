"""create_database — single entry point for building a Database from config."""

from __future__ import annotations

import importlib

from .backends.mysql import MySQLGateway
from .backends.null import NullGateway
from .backends.sqlite import SqliteGateway
from .config import DatabaseConfig
from .gateway import Database

_KNOWN_BACKENDS = ("null", "postgres", "mysql", "sqlite")


def create_database(config: DatabaseConfig) -> Database:
    """Dispatch on `config.backend`.

    - "null"     → NullGateway
    - "postgres" → PostgresGateway (raises ImportError if asyncpg missing)
    - "mysql"    → NotImplementedError (placeholder)
    - "sqlite"   → NotImplementedError (placeholder)
    - other      → ValueError
    """
    backend = config.backend
    if backend == "null":
        return NullGateway()
    if backend == "postgres":
        try:
            mod = importlib.import_module(
                "topsport_agent.database.backends.postgres"
            )
            return mod.PostgresGateway(config)
        except ImportError as exc:
            raise ImportError(
                "PostgresGateway requires 'asyncpg'. Install via: "
                "uv sync --group db"
            ) from exc
    if backend == "mysql":
        return MySQLGateway(config)
    if backend == "sqlite":
        return SqliteGateway(config)
    raise ValueError(
        f"unknown backend: {backend!r}. Known: {_KNOWN_BACKENDS}"
    )


__all__ = ["create_database"]
