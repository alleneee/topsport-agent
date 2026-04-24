"""Protocol surface for pluggable databases. Backend-agnostic.

Design decisions (see spec §5.2):
- Named parameter style (":tenant_id") is canonical; backends translate.
- Row type is Mapping[str, Any] — no schema binding.
- Transaction is context-manager-only; no begin/commit/rollback.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Transaction(Protocol):
    """A scoped transactional view of the Database.

    Exceptions escaping the `async with db.transaction() as tx:` block cause
    rollback. Normal exit commits. No explicit commit/rollback methods are
    exposed.
    """

    async def execute(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> int: ...

    async def fetch_one(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any] | None: ...

    async def fetch_all(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> list[Mapping[str, Any]]: ...

    async def fetch_val(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Any: ...


@runtime_checkable
class Database(Protocol):
    """Top-level pluggable database gateway.

    Invariants:
    - connect() / close() are idempotent.
    - health_check() returns True on a healthy pool, False otherwise (no raise).
    - execute / fetch_* use named parameters (":name"); backends translate.
    """

    @property
    def dialect(self) -> str: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def health_check(self) -> bool: ...

    async def execute(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> int: ...

    async def fetch_one(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any] | None: ...

    async def fetch_all(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> list[Mapping[str, Any]]: ...

    async def fetch_val(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Any: ...

    def transaction(self) -> AbstractAsyncContextManager[Transaction]: ...


__all__ = ["Database", "Transaction"]
