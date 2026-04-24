"""NullGateway — used when `enable_database=False`.

Lifecycle methods are no-ops; any query attempt raises RuntimeError so an
accidentally-wired store crashes loudly instead of silently doing nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

_DISABLED_MSG = "database disabled (enable_database=False or backend='null')"


class NullGateway:
    @property
    def dialect(self) -> str:
        return "null"

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health_check(self) -> bool:
        return True

    async def execute(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> int:
        raise RuntimeError(_DISABLED_MSG)

    async def fetch_one(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any] | None:
        raise RuntimeError(_DISABLED_MSG)

    async def fetch_all(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> list[Mapping[str, Any]]:
        raise RuntimeError(_DISABLED_MSG)

    async def fetch_val(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Any:
        raise RuntimeError(_DISABLED_MSG)

    def transaction(self):
        @asynccontextmanager
        async def _raise():
            raise RuntimeError(_DISABLED_MSG)
            yield  # unreachable; satisfies the generator protocol

        return _raise()


__all__ = ["NullGateway"]
