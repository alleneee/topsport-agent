"""DatabaseConfig — pool/connection parameters, frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DatabaseConfig:
    """Configuration for the Database Gateway.

    backend: "null" | "postgres" | "mysql" | "sqlite"
    url:     dsn in the native format of the chosen backend
             (postgresql://user:pw@host:5432/db for Postgres)
    """

    backend: str = "null"
    url: str | None = None
    pool_min: int = 1
    pool_max: int = 10
    timeout_seconds: float = 30.0


__all__ = ["DatabaseConfig"]
