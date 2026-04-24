"""PostgresGateway — asyncpg-backed pool lifecycle. Query methods deliberately
raise NotImplementedError to signal "configured but not yet wired for writes".

The module imports without asyncpg installed — the ImportError is deferred to
construction time via importlib.import_module(variable).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from ..config import DatabaseConfig
from ..errors import ConnectionError as DBConnectionError


class PostgresGateway:
    """Skeleton: pool lifecycle works; query methods NotImplementedError.

    Deliberate choice — downstream store implementations will fill these in
    incrementally. Until then, any accidental use crashes loudly.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._pool: Any | None = None  # asyncpg.Pool when connected
        # Indirect import so Pyright + environments without asyncpg don't choke.
        mod_name = "asyncpg"
        self._asyncpg = importlib.import_module(mod_name)

    @property
    def dialect(self) -> str:
        return "postgres"

    async def connect(self) -> None:
        if self._pool is not None:
            return  # idempotent
        if not self._config.url:
            raise DBConnectionError("DatabaseConfig.url is empty for postgres backend")
        try:
            self._pool = await self._asyncpg.create_pool(
                dsn=self._config.url,
                min_size=self._config.pool_min,
                max_size=self._config.pool_max,
                timeout=self._config.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 — translate driver errors
            raise DBConnectionError(f"postgres connect failed: {exc!r}") from exc

    async def close(self) -> None:
        if self._pool is None:
            return
        pool = self._pool
        self._pool = None
        try:
            await pool.close()
        except Exception:  # noqa: BLE001 — close should never raise
            pass

    async def health_check(self) -> bool:
        if self._pool is None:
            return False
        try:
            async with self._pool.acquire() as conn:
                value = await conn.fetchval("SELECT 1")
            return value == 1
        except Exception:  # noqa: BLE001
            return False

    async def execute(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> int:
        raise NotImplementedError(
            "PostgresGateway.execute() not implemented yet — query layer pending"
        )

    async def fetch_one(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any] | None:
        raise NotImplementedError(
            "PostgresGateway.fetch_one() not implemented yet — query layer pending"
        )

    async def fetch_all(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> list[Mapping[str, Any]]:
        raise NotImplementedError(
            "PostgresGateway.fetch_all() not implemented yet — query layer pending"
        )

    async def fetch_val(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Any:
        raise NotImplementedError(
            "PostgresGateway.fetch_val() not implemented yet — query layer pending"
        )

    def transaction(self):
        @asynccontextmanager
        async def _raise():
            raise NotImplementedError(
                "PostgresGateway.transaction() not implemented yet — query layer pending"
            )
            yield  # unreachable

        return _raise()


__all__ = ["PostgresGateway"]
