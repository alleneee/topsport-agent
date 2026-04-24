"""Placeholder — MySQL backend slot reserved. Raises on construction."""

from __future__ import annotations

from ..config import DatabaseConfig


class MySQLGateway:
    def __init__(self, config: DatabaseConfig) -> None:
        raise NotImplementedError("mysql backend not implemented yet")


__all__ = ["MySQLGateway"]
