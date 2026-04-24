# Database Abstraction Skeleton — Implementation Plan (Plan A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pluggable `Database` Gateway abstraction with 4 backends (null/postgres/mysql/sqlite) where only Postgres has real pool-lifecycle code; all query methods remain `NotImplementedError`. Wire into `ServerConfig` + FastAPI lifespan so `ENABLE_DATABASE=true` is a working, health-checkable code path.

**Architecture:** `Database` Protocol + `Transaction` Protocol in `src/topsport_agent/database/`. Backends live in `database/backends/`. Factory dispatches on `config.backend`. Import style follows the project's `importlib.import_module(variable)` pattern so the module imports cleanly without `asyncpg` installed.

**Tech Stack:** Python 3.11, `asyncpg` (optional), pytest-asyncio, FastAPI lifespan, dataclasses-only (no pydantic for config).

**Related spec:** `docs/superpowers/specs/2026-04-24-database-abstraction-and-redis-ratelimit-design.md` sections 3, 4, 5, 7.

---

## Task 0: Scaffold module + add `db` dependency group

**Files:**
- Modify: `pyproject.toml` (add `db` group)
- Create: `src/topsport_agent/database/__init__.py` (empty placeholder)
- Create: `src/topsport_agent/database/backends/__init__.py` (empty placeholder)
- Test: `tests/test_database_scaffold.py`

- [ ] **Step 1: Write failing test asserting the module is importable**

```python
# tests/test_database_scaffold.py
def test_database_module_imports() -> None:
    import topsport_agent.database  # noqa: F401
    import topsport_agent.database.backends  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_database_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'topsport_agent.database'`

- [ ] **Step 3: Add `db` dependency group to `pyproject.toml`**

Find the `[dependency-groups]` block and add (after existing `browser` group):

```toml
db = [
    "asyncpg>=0.29.0",
]
```

- [ ] **Step 4: Create two empty `__init__.py` files**

```python
# src/topsport_agent/database/__init__.py
"""Database Gateway abstraction — pluggable multi-backend skeleton."""
```

```python
# src/topsport_agent/database/backends/__init__.py
"""Concrete Database backends (null / postgres / mysql / sqlite)."""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_database_scaffold.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/topsport_agent/database/ tests/test_database_scaffold.py
git commit -m "feat(database): scaffold module + db dependency group"
```

---

## Task 1: Error hierarchy (`errors.py`)

**Files:**
- Create: `src/topsport_agent/database/errors.py`
- Test: `tests/test_database_errors.py`

- [ ] **Step 1: Write failing test for error hierarchy**

```python
# tests/test_database_errors.py
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
```

- [ ] **Step 2: Run test, expect FAIL with `ModuleNotFoundError`**

Run: `uv run pytest tests/test_database_errors.py -v`

- [ ] **Step 3: Implement `errors.py`**

```python
# src/topsport_agent/database/errors.py
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
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/test_database_errors.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/database/errors.py tests/test_database_errors.py
git commit -m "feat(database): error hierarchy"
```

---

## Task 2: `DatabaseConfig` dataclass (`config.py`)

**Files:**
- Create: `src/topsport_agent/database/config.py`
- Test: `tests/test_database_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_database_config.py
from topsport_agent.database.config import DatabaseConfig


def test_default_config_is_null_backend() -> None:
    cfg = DatabaseConfig()
    assert cfg.backend == "null"
    assert cfg.url is None
    assert cfg.pool_min == 1
    assert cfg.pool_max == 10
    assert cfg.timeout_seconds == 30.0


def test_config_is_frozen() -> None:
    import dataclasses

    cfg = DatabaseConfig()
    with pytest_raises_frozen():
        cfg.backend = "postgres"  # type: ignore[misc]


def pytest_raises_frozen():
    import dataclasses
    import pytest

    return pytest.raises(dataclasses.FrozenInstanceError)


def test_custom_config_round_trip() -> None:
    cfg = DatabaseConfig(
        backend="postgres",
        url="postgresql://localhost/test",
        pool_min=2,
        pool_max=20,
        timeout_seconds=5.0,
    )
    assert cfg.backend == "postgres"
    assert cfg.pool_max == 20
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_database_config.py -v`

- [ ] **Step 3: Implement `config.py`**

```python
# src/topsport_agent/database/config.py
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
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/test_database_config.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/database/config.py tests/test_database_config.py
git commit -m "feat(database): DatabaseConfig frozen dataclass"
```

---

## Task 3: `Database` / `Transaction` Protocols (`gateway.py`)

Protocols are structural; the test here is "any class satisfying the Protocol can be assigned to a `Database`-typed variable". We defer runtime conformance testing until Task 4 (NullGateway).

**Files:**
- Create: `src/topsport_agent/database/gateway.py`
- Test: `tests/test_database_gateway.py`

- [ ] **Step 1: Write failing import test**

```python
# tests/test_database_gateway.py
def test_gateway_protocols_importable() -> None:
    from topsport_agent.database.gateway import Database, Transaction  # noqa: F401


def test_gateway_has_required_attrs() -> None:
    from topsport_agent.database.gateway import Database, Transaction

    for proto in (Database, Transaction):
        # Protocol attributes live on __annotations__ / methods on the class body
        names = set(dir(proto))
        assert "execute" in names
        assert "fetch_one" in names
        assert "fetch_all" in names
        assert "fetch_val" in names

    # Only Database has lifecycle + dialect + transaction factory
    db_names = set(dir(Database))
    for required in ("dialect", "connect", "close", "health_check", "transaction"):
        assert required in db_names, required
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_database_gateway.py -v`

- [ ] **Step 3: Implement `gateway.py`**

```python
# src/topsport_agent/database/gateway.py
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
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/test_database_gateway.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/database/gateway.py tests/test_database_gateway.py
git commit -m "feat(database): Database + Transaction Protocols"
```

---

## Task 4: `NullGateway` backend (`backends/null.py`)

**Files:**
- Create: `src/topsport_agent/database/backends/null.py`
- Test: `tests/test_database_null_backend.py`

- [ ] **Step 1: Write failing tests for NullGateway behavior**

```python
# tests/test_database_null_backend.py
import pytest

from topsport_agent.database.backends.null import NullGateway
from topsport_agent.database.gateway import Database


def test_null_gateway_satisfies_database_protocol() -> None:
    gw = NullGateway()
    assert isinstance(gw, Database)  # runtime_checkable Protocol


def test_dialect_is_null() -> None:
    assert NullGateway().dialect == "null"


@pytest.mark.asyncio
async def test_lifecycle_is_noop() -> None:
    gw = NullGateway()
    await gw.connect()
    await gw.connect()  # idempotent
    assert await gw.health_check() is True
    await gw.close()
    await gw.close()  # idempotent


@pytest.mark.asyncio
async def test_queries_raise_disabled_error() -> None:
    gw = NullGateway()
    with pytest.raises(RuntimeError, match="database disabled"):
        await gw.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="database disabled"):
        await gw.fetch_one("SELECT 1")
    with pytest.raises(RuntimeError, match="database disabled"):
        await gw.fetch_all("SELECT 1")
    with pytest.raises(RuntimeError, match="database disabled"):
        await gw.fetch_val("SELECT 1")


@pytest.mark.asyncio
async def test_transaction_raises_disabled_error() -> None:
    gw = NullGateway()
    with pytest.raises(RuntimeError, match="database disabled"):
        async with gw.transaction():
            pass
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/test_database_null_backend.py -v`

- [ ] **Step 3: Implement `NullGateway`**

```python
# src/topsport_agent/database/backends/null.py
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
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `uv run pytest tests/test_database_null_backend.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/database/backends/null.py tests/test_database_null_backend.py
git commit -m "feat(database): NullGateway — default no-op backend"
```

---

## Task 5: MySQL + SQLite placeholder backends

Both placeholders fail at construction so no one accidentally uses them.

**Files:**
- Create: `src/topsport_agent/database/backends/mysql.py`
- Create: `src/topsport_agent/database/backends/sqlite.py`
- Test: `tests/test_database_placeholder_backends.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_database_placeholder_backends.py
import pytest

from topsport_agent.database.backends.mysql import MySQLGateway
from topsport_agent.database.backends.sqlite import SqliteGateway
from topsport_agent.database.config import DatabaseConfig


def test_mysql_gateway_raises_on_construction() -> None:
    with pytest.raises(NotImplementedError, match="mysql backend not implemented"):
        MySQLGateway(DatabaseConfig(backend="mysql", url="mysql://x"))


def test_sqlite_gateway_raises_on_construction() -> None:
    with pytest.raises(NotImplementedError, match="sqlite backend not implemented"):
        SqliteGateway(DatabaseConfig(backend="sqlite", url="sqlite:///x"))
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/test_database_placeholder_backends.py -v`

- [ ] **Step 3: Implement both placeholders**

```python
# src/topsport_agent/database/backends/mysql.py
"""Placeholder — MySQL backend slot reserved. Raises on construction."""

from __future__ import annotations

from ..config import DatabaseConfig


class MySQLGateway:
    def __init__(self, config: DatabaseConfig) -> None:
        raise NotImplementedError("mysql backend not implemented yet")


__all__ = ["MySQLGateway"]
```

```python
# src/topsport_agent/database/backends/sqlite.py
"""Placeholder — SQLite backend slot reserved (future: aiosqlite). Raises on construction."""

from __future__ import annotations

from ..config import DatabaseConfig


class SqliteGateway:
    def __init__(self, config: DatabaseConfig) -> None:
        raise NotImplementedError("sqlite backend not implemented yet")


__all__ = ["SqliteGateway"]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `uv run pytest tests/test_database_placeholder_backends.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/database/backends/mysql.py src/topsport_agent/database/backends/sqlite.py tests/test_database_placeholder_backends.py
git commit -m "feat(database): mysql + sqlite placeholder backends"
```

---

## Task 6: `PostgresGateway` — pool lifecycle skeleton

The non-trivial task. Real `asyncpg.create_pool` wiring, but query methods raise `NotImplementedError`. Tests run without a live Postgres; they only validate (a) failing connect → our `ConnectionError`, (b) queries raise `NotImplementedError`, (c) imports work without asyncpg (skip gracefully).

**Files:**
- Create: `src/topsport_agent/database/backends/postgres.py`
- Test: `tests/test_database_backends_postgres.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_database_backends_postgres.py
import importlib

import pytest

from topsport_agent.database.config import DatabaseConfig
from topsport_agent.database.errors import ConnectionError as DBConnectionError

_asyncpg_available = importlib.util.find_spec("asyncpg") is not None
requires_asyncpg = pytest.mark.skipif(
    not _asyncpg_available, reason="asyncpg not installed (uv sync --group db)"
)


@requires_asyncpg
def test_postgres_gateway_dialect() -> None:
    from topsport_agent.database.backends.postgres import PostgresGateway

    gw = PostgresGateway(DatabaseConfig(backend="postgres", url="postgresql://x"))
    assert gw.dialect == "postgres"


@requires_asyncpg
@pytest.mark.asyncio
async def test_postgres_gateway_connect_failure_wraps_as_db_connection_error() -> None:
    """Unreachable host → our ConnectionError, not raw asyncpg exception."""
    from topsport_agent.database.backends.postgres import PostgresGateway

    gw = PostgresGateway(
        DatabaseConfig(
            backend="postgres",
            url="postgresql://nonexistent.invalid:5432/db",
            timeout_seconds=1.0,
        )
    )
    with pytest.raises(DBConnectionError):
        await gw.connect()


@requires_asyncpg
@pytest.mark.asyncio
async def test_postgres_gateway_queries_raise_not_implemented() -> None:
    """Until real SQL is wired, every query method explicitly errors out."""
    from topsport_agent.database.backends.postgres import PostgresGateway

    gw = PostgresGateway(DatabaseConfig(backend="postgres", url="postgresql://x"))
    with pytest.raises(NotImplementedError):
        await gw.execute("SELECT 1")
    with pytest.raises(NotImplementedError):
        await gw.fetch_one("SELECT 1")
    with pytest.raises(NotImplementedError):
        await gw.fetch_all("SELECT 1")
    with pytest.raises(NotImplementedError):
        await gw.fetch_val("SELECT 1")
    with pytest.raises(NotImplementedError):
        async with gw.transaction():
            pass


@requires_asyncpg
@pytest.mark.asyncio
async def test_postgres_gateway_close_is_idempotent_before_connect() -> None:
    """close() before connect() must not raise; required for error paths in lifespan."""
    from topsport_agent.database.backends.postgres import PostgresGateway

    gw = PostgresGateway(DatabaseConfig(backend="postgres", url="postgresql://x"))
    await gw.close()
    await gw.close()  # twice, still fine


def test_postgres_gateway_importable_without_asyncpg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module imports even without asyncpg — the hard dependency kicks in at construction."""
    import sys

    # Even if asyncpg isn't installed in the dev env running this test,
    # the module-level import of postgres.py should succeed.
    mod = importlib.import_module("topsport_agent.database.backends.postgres")
    assert hasattr(mod, "PostgresGateway")
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/test_database_backends_postgres.py -v`

- [ ] **Step 3: Implement `PostgresGateway` skeleton**

```python
# src/topsport_agent/database/backends/postgres.py
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
```

- [ ] **Step 4: Run tests, expect PASS (requires `--group db` for real asyncpg tests to run; other tests run always)**

Run: `uv sync --group db && uv run pytest tests/test_database_backends_postgres.py -v`

Expected: all PASS. If `asyncpg` unavailable, `@requires_asyncpg`-marked tests skip and the non-marked "importable" test still passes.

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/database/backends/postgres.py tests/test_database_backends_postgres.py
git commit -m "feat(database): PostgresGateway pool lifecycle skeleton"
```

---

## Task 7: `create_database` factory (`factory.py`)

**Files:**
- Create: `src/topsport_agent/database/factory.py`
- Test: `tests/test_database_factory.py`
- Modify: `src/topsport_agent/database/__init__.py` (re-export public surface)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_database_factory.py
import importlib

import pytest

from topsport_agent.database.backends.null import NullGateway
from topsport_agent.database.config import DatabaseConfig
from topsport_agent.database.factory import create_database

_asyncpg_available = importlib.util.find_spec("asyncpg") is not None


def test_factory_null_backend_returns_null_gateway() -> None:
    db = create_database(DatabaseConfig(backend="null"))
    assert isinstance(db, NullGateway)


def test_factory_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        create_database(DatabaseConfig(backend="mongodb"))  # type: ignore[arg-type]


def test_factory_mysql_placeholder_raises() -> None:
    with pytest.raises(NotImplementedError, match="mysql backend not implemented"):
        create_database(DatabaseConfig(backend="mysql"))


def test_factory_sqlite_placeholder_raises() -> None:
    with pytest.raises(NotImplementedError, match="sqlite backend not implemented"):
        create_database(DatabaseConfig(backend="sqlite"))


@pytest.mark.skipif(not _asyncpg_available, reason="asyncpg not installed")
def test_factory_postgres_returns_postgres_gateway() -> None:
    from topsport_agent.database.backends.postgres import PostgresGateway

    db = create_database(
        DatabaseConfig(backend="postgres", url="postgresql://x")
    )
    assert isinstance(db, PostgresGateway)


@pytest.mark.skipif(_asyncpg_available, reason="requires asyncpg to be MISSING")
def test_factory_postgres_missing_asyncpg_raises_import_error() -> None:
    """When asyncpg is not installed, selecting postgres must raise with a helpful message."""
    with pytest.raises(ImportError, match="asyncpg"):
        create_database(DatabaseConfig(backend="postgres", url="postgresql://x"))


def test_public_package_exports() -> None:
    import topsport_agent.database as db_pkg

    assert hasattr(db_pkg, "create_database")
    assert hasattr(db_pkg, "DatabaseConfig")
    assert hasattr(db_pkg, "Database")
    assert hasattr(db_pkg, "NullGateway")
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/test_database_factory.py -v`

- [ ] **Step 3: Implement `factory.py`**

```python
# src/topsport_agent/database/factory.py
"""create_database — single entry point for building a Database from config."""

from __future__ import annotations

from .backends.mysql import MySQLGateway
from .backends.null import NullGateway
from .backends.sqlite import SqliteGateway
from .config import DatabaseConfig
from .gateway import Database

_KNOWN_BACKENDS = ("null", "postgres", "mysql", "sqlite")


def create_database(config: DatabaseConfig) -> Database:
    """Dispatch on `config.backend`.

    - "null"       → NullGateway
    - "postgres"   → PostgresGateway (raises ImportError if asyncpg missing)
    - "mysql"      → NotImplementedError (placeholder)
    - "sqlite"     → NotImplementedError (placeholder)
    - anything else → ValueError
    """
    backend = config.backend
    if backend == "null":
        return NullGateway()
    if backend == "postgres":
        # Import lazily so users without --group db can still import this module.
        import importlib

        try:
            mod = importlib.import_module(
                "topsport_agent.database.backends.postgres"
            )
        except ImportError as exc:
            raise ImportError(
                "PostgresGateway requires 'asyncpg'. Install via: "
                "uv sync --group db"
            ) from exc
        return mod.PostgresGateway(config)
    if backend == "mysql":
        return MySQLGateway(config)
    if backend == "sqlite":
        return SqliteGateway(config)
    raise ValueError(
        f"unknown backend: {backend!r}. Known: {_KNOWN_BACKENDS}"
    )


__all__ = ["create_database"]
```

- [ ] **Step 4: Update package `__init__.py`**

```python
# src/topsport_agent/database/__init__.py
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
```

- [ ] **Step 5: Run tests, expect PASS**

Run: `uv run pytest tests/test_database_factory.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/topsport_agent/database/factory.py src/topsport_agent/database/__init__.py tests/test_database_factory.py
git commit -m "feat(database): create_database factory + public exports"
```

---

## Task 8: Extend `ServerConfig` with database fields

**Files:**
- Modify: `src/topsport_agent/server/config.py` (add fields + env parsing)
- Test: `tests/test_server_config_database.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_server_config_database.py
import pytest

from topsport_agent.server.config import ServerConfig


def test_defaults_disable_database() -> None:
    cfg = ServerConfig()
    assert cfg.enable_database is False
    assert cfg.database_backend == "postgres"  # the "default when enabled" value
    assert cfg.database_url is None
    assert cfg.database_pool_min == 1
    assert cfg.database_pool_max == 10
    assert cfg.database_timeout_seconds == 30.0


def test_env_parsing_enable_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_DATABASE", "true")
    monkeypatch.setenv("DATABASE_BACKEND", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/x")
    monkeypatch.setenv("DATABASE_POOL_MIN", "2")
    monkeypatch.setenv("DATABASE_POOL_MAX", "5")
    # api_key is required for from_env path not to fail; set a dummy.
    monkeypatch.setenv("API_KEY", "dummy")

    cfg = ServerConfig.from_env()
    assert cfg.enable_database is True
    assert cfg.database_backend == "postgres"
    assert cfg.database_url == "postgresql://localhost/x"
    assert cfg.database_pool_min == 2
    assert cfg.database_pool_max == 5


def test_env_parsing_database_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Make sure no stray env vars leak in.
    for k in (
        "ENABLE_DATABASE", "DATABASE_BACKEND", "DATABASE_URL",
        "DATABASE_POOL_MIN", "DATABASE_POOL_MAX", "DATABASE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("API_KEY", "dummy")

    cfg = ServerConfig.from_env()
    assert cfg.enable_database is False
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `uv run pytest tests/test_server_config_database.py -v`

- [ ] **Step 3: Add fields and env parsing to `server/config.py`**

Edit `src/topsport_agent/server/config.py`. In the `ServerConfig` dataclass, add these fields right before the closing `@classmethod from_env` (so after `sandbox_use_server_proxy`):

```python
    # Database (pluggable skeleton; see database/)
    enable_database: bool = False
    database_backend: str = "postgres"          # only applied when enable_database=True
    database_url: str | None = None
    database_pool_min: int = 1
    database_pool_max: int = 10
    database_timeout_seconds: float = 30.0
```

Then in the `from_env` classmethod, extend the `cls(...)` call (before the closing paren) with:

```python
            enable_database=_parse_bool(
                os.environ.get("ENABLE_DATABASE"), default=False
            ),
            database_backend=os.environ.get("DATABASE_BACKEND", "postgres"),
            database_url=os.environ.get("DATABASE_URL") or None,
            database_pool_min=int(os.environ.get("DATABASE_POOL_MIN", "1")),
            database_pool_max=int(os.environ.get("DATABASE_POOL_MAX", "10")),
            database_timeout_seconds=float(
                os.environ.get("DATABASE_TIMEOUT_SECONDS", "30")
            ),
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `uv run pytest tests/test_server_config_database.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/topsport_agent/server/config.py tests/test_server_config_database.py
git commit -m "feat(server): ServerConfig database fields + env parsing"
```

---

## Task 9: Lifespan integration in `server/app.py`

**Files:**
- Modify: `src/topsport_agent/server/app.py` (lifespan setup + teardown)
- Test: `tests/test_server_lifespan_database.py`

- [ ] **Step 1: Write failing test for lifespan wiring**

```python
# tests/test_server_lifespan_database.py
"""Verify ENABLE_DATABASE flag drives lifespan wiring.

These tests don't require a running Postgres — they test the control flow:
- disabled → app.state.database is NullGateway
- enabled + bad URL → lifespan raises at startup
"""
from __future__ import annotations

import pytest

from topsport_agent.database.backends.null import NullGateway
from topsport_agent.server.app import create_app
from topsport_agent.server.config import ServerConfig


@pytest.mark.asyncio
async def test_lifespan_disabled_database_uses_null_gateway() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_database=False,
    )
    app = create_app(cfg, provider=_make_stub_provider())
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.database, NullGateway)
        assert await app.state.database.health_check() is True


@pytest.mark.asyncio
async def test_lifespan_enabled_without_url_raises() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_database=True,
        database_backend="postgres",
        database_url=None,
    )
    app = create_app(cfg, provider=_make_stub_provider())
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        async with app.router.lifespan_context(app):
            pass


def _make_stub_provider():
    class _Prov:
        name = "stub"

        async def complete(self, request):  # pragma: no cover
            raise RuntimeError("not used in lifespan tests")

    return _Prov()
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `uv run pytest tests/test_server_lifespan_database.py -v`

- [ ] **Step 3: Modify `server/app.py` lifespan**

First, add an import near the top of `server/app.py` (after the existing `from .config import ServerConfig`):

```python
from ..database import NullGateway, create_database
from ..database.config import DatabaseConfig
```

Then **inside `create_app`'s lifespan function**, after the existing
`app.state.config = cfg` / `app.state.auth_config = auth_config` block and
**before** the `if not auth_config.required:` warning, insert the database
wiring block:

```python
        # === Database (optional, default off) ===
        if cfg.enable_database:
            if not cfg.database_url:
                raise RuntimeError(
                    "ENABLE_DATABASE=true but DATABASE_URL is unset"
                )
            db_config = DatabaseConfig(
                backend=cfg.database_backend,
                url=cfg.database_url,
                pool_min=cfg.database_pool_min,
                pool_max=cfg.database_pool_max,
                timeout_seconds=cfg.database_timeout_seconds,
            )
            db = create_database(db_config)
            await db.connect()
            if not await db.health_check():
                raise RuntimeError(
                    "database enabled but health_check failed"
                )
            app.state.database = db
        else:
            app.state.database = NullGateway()
```

Then extend the **finally** block of the lifespan to close the database.
Find the existing `finally:` clause (around the drain logic near line 241)
and add **right after** the drain handling (i.e., at the very end of the
finally block, just before the lifespan function returns):

```python
            # Close database if it was a real backend (NullGateway.close is no-op).
            db_state = getattr(app.state, "database", None)
            if db_state is not None:
                try:
                    await db_state.close()
                except Exception:  # noqa: BLE001
                    _logger.warning(
                        "database close failed during shutdown", exc_info=True
                    )
```

- [ ] **Step 4: Run test, expect PASS**

Run: `uv run pytest tests/test_server_lifespan_database.py -v`

- [ ] **Step 5: Run the whole suite to ensure nothing regressed**

Run: `uv run pytest -v`

Expected: all previously green tests still green; new tests green.

- [ ] **Step 6: Commit**

```bash
git add src/topsport_agent/server/app.py tests/test_server_lifespan_database.py
git commit -m "feat(server): database lifespan wiring"
```

---

## Task 10: README documentation

**Files:**
- Modify: `README.md` (add a Database section)

- [ ] **Step 1: Locate the right spot**

Scan `README.md` for the existing module table or the "Configuration" section. Insert the new section after the existing configuration documentation and before any "Testing" section. If the project has a "Status" block with module names, append "Database (skeleton)" to it.

- [ ] **Step 2: Add the Database section**

Append the following markdown (replace any existing "Database" stub first):

```markdown
## Database Abstraction (skeleton)

Pluggable multi-backend database gateway. The abstraction is fully landed;
backends ship with **pool lifecycle only** — query methods raise
`NotImplementedError` until downstream stores wire them in (separate spec).

### Enabling

```bash
uv sync --group db                       # install asyncpg
export ENABLE_DATABASE=true
export DATABASE_BACKEND=postgres         # default when enabled
export DATABASE_URL="postgresql://user:pw@localhost:5432/mydb"
```

### Supported backends

| Backend    | Status                                           |
| ---------- | ------------------------------------------------ |
| `null`     | Default when `ENABLE_DATABASE=false`. No-op.     |
| `postgres` | Pool lifecycle + `health_check` implemented.     |
| `mysql`    | Placeholder — raises `NotImplementedError`.      |
| `sqlite`   | Placeholder — raises `NotImplementedError`.      |

### Environment variables

| Variable                      | Default    | Notes                                   |
| ----------------------------- | ---------- | --------------------------------------- |
| `ENABLE_DATABASE`             | `false`    | Master switch                           |
| `DATABASE_BACKEND`            | `postgres` | Only used when enabled                  |
| `DATABASE_URL`                | —          | Required when enabled                   |
| `DATABASE_POOL_MIN`           | `1`        |                                         |
| `DATABASE_POOL_MAX`           | `10`       |                                         |
| `DATABASE_TIMEOUT_SECONDS`    | `30`       | Connect timeout                         |

### Behavior

- `ENABLE_DATABASE=false` → `app.state.database` is a `NullGateway`; imports
  succeed even without `asyncpg` installed.
- `ENABLE_DATABASE=true` + unreachable Postgres → server **fails to start**
  (fail-fast; prevents serving with a broken dependency).
- `ENABLE_DATABASE=true` + good URL → `health_check` reports True;
  `execute()` / `fetch_*()` still raise `NotImplementedError` (deliberate —
  store implementations arrive in a separate spec).
```

- [ ] **Step 3: Lint markdown**

Run: `npx markdownlint-cli2 README.md` (if the project has it set up) OR manually verify with a preview.

If a markdownlint config exists in repo root and reports issues, fix them.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(database): README section for the pluggable skeleton"
```

---

## Task 11: Final verification

- [ ] **Step 1: Full test suite without optional groups**

Run: `uv sync && uv run pytest -v`

Expected: all green. Tests marked `@requires_asyncpg` should **skip** gracefully.

- [ ] **Step 2: Full test suite with `--group db`**

Run: `uv sync --group db --group dev && uv run pytest -v`

Expected: all green, including the `PostgresGateway` tests that need asyncpg.

- [ ] **Step 3: Verify no direct `asyncpg` import leaks into top-level modules**

Run:

```bash
grep -rn "^import asyncpg\|^from asyncpg" src/topsport_agent/
```

Expected: **no output**. The only reference must be `importlib.import_module("asyncpg")` inside `backends/postgres.py`.

- [ ] **Step 4: Verify the package imports without asyncpg**

```bash
uv run python -c "from topsport_agent.database import create_database, Database, NullGateway; print('OK')"
```

Expected: prints `OK`. This must succeed **even without** `--group db` synced.

- [ ] **Step 5: Manual acceptance checklist from spec §10**

Go through each checkbox in `docs/superpowers/specs/2026-04-24-database-abstraction-and-redis-ratelimit-design.md` §10.4 (Database), manually verify each passes.

- [ ] **Step 6: Final commit + tag (optional)**

If all green:

```bash
git log --oneline -15                    # sanity-check the task commits
# no extra commit needed; the per-task commits are the ship record
```

---

## Self-Review Notes

**Spec coverage** (cross-check §3/§5/§7 of the spec):

- §3 directory layout → Task 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 (all 10 source files)
- §4 invariants → Task 11 (acceptance) + Task 6 (`asyncpg` not leaked) + Task 9 (default off path)
- §5.1 Protocol → Task 3
- §5.2 design decisions → codified in code comments (Task 3 docstring)
- §5.3 backend completeness table → Task 4 (null) / Task 5 (mysql+sqlite) / Task 6 (postgres)
- §5.4 errors → Task 1
- §5.5 config & factory → Task 2 (config) + Task 7 (factory)
- §7 ServerConfig + lifespan → Task 8 + Task 9

**No placeholders**: scanned — every task has concrete code blocks, file paths, commands, and expected outputs.

**Type consistency**: `Database` / `Transaction` / `NullGateway` / `PostgresGateway` / `DatabaseConfig` / `create_database` — all names stable across tasks. `ConnectionError` in `database.errors` intentionally shadows the builtin; always imported as `from ..errors import ConnectionError as DBConnectionError` when needed at call sites.
