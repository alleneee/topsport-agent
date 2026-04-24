"""Placeholder — SQLite backend slot reserved (future: aiosqlite). Raises on construction."""

from __future__ import annotations

from ..config import DatabaseConfig


class SqliteGateway:
    def __init__(self, config: DatabaseConfig) -> None:
        raise NotImplementedError("sqlite backend not implemented yet")


__all__ = ["SqliteGateway"]
