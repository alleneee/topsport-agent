# Database Abstraction + Redis Rate Limiting Design

- Date: 2026-04-24
- Status: Proposed
- Author: niko (with brainstorming via Claude)

## 1. Context

`topsport-agent` currently has **no database layer** and **no rate limiting**.
Several storage Protocols (`MemoryStore`, `AuditStore`, `PersonaRegistry`,
`AssignmentStore`) already have in-memory / file-backed implementations —
natural plug-in points for a SQL backend — but there is no shared abstraction
for database access, and every production deployment would duplicate
connection-pool wiring per store.

The server is a FastAPI app with auth, RBAC, and capability-ACL already
landed, but no way to throttle requests by IP / principal / tenant / route.
An enterprise internal AI platform needs rate limiting for tenant billing
alignment, abuse prevention, and fair-use enforcement.

This spec delivers two **parallel, independent** subsystems:

1. **Database abstraction layer** — a `Database` Gateway Protocol with
   multi-backend plug-in architecture, **skeleton only** (no SQL implemented).
2. **Redis-backed rate limiting** — a complete, fully-implemented middleware
   with sliding-window semantics across four dimensions (IP / principal /
   tenant / route).

The two subsystems do **not** depend on each other at runtime. They are
grouped in one spec because they share: optional-dependency packaging style,
default-off enable switches, and the same deployment/ops onboarding surface.

## 2. Non-Goals

Explicitly **out of scope** to keep this spec focused:

- Any SQL queries / schema / migrations — database is skeleton-only
- Connecting any existing store (`MemoryStore` / `AuditStore` / etc.) to the
  new `Database` — that's the follow-up spec
- `SessionStore` persistence, crash recovery, or SSE stream resume
- `KillSwitch` cross-replica broadcast via Redis
- Token-bucket algorithm (sliding-window is sufficient for v1)
- Per-tool rate limits (tool population is dynamic; deferred to v2)
- Dynamic rate-limit configuration (admin API to change quotas at runtime)
- `/health` endpoint enrichment with DB / Redis status
- CI workflow YAML changes (only README guidance)
- MySQL / SQLite real implementations (placeholder only)

## 3. Architecture Overview

```
src/topsport_agent/
├── database/                  (NEW) pluggable DB abstraction, skeleton only
│   ├── gateway.py             Database / Transaction Protocol
│   ├── config.py              DatabaseConfig
│   ├── factory.py             create_database(config) -> Database
│   ├── errors.py              DatabaseError / ConnectionError / ...
│   └── backends/
│       ├── null.py            NullGateway (default when disabled)
│       ├── postgres.py        PostgresGateway (pool lifecycle only)
│       ├── mysql.py           placeholder: raise NotImplementedError
│       └── sqlite.py          placeholder: raise NotImplementedError
│
├── ratelimit/                 (NEW) complete Redis-backed rate limiting
│   ├── types.py               RateLimitScope / Rule / Decision
│   ├── limiter.py             RateLimiter Protocol + RedisSlidingWindowLimiter
│   ├── redis_client.py        create_redis_client(url)
│   ├── middleware.py          RateLimitMiddleware (FastAPI)
│   ├── metrics.py             Prometheus counters
│   ├── config.py              RateLimitConfig
│   └── lua/sliding_window.lua atomic multi-dimension limiter script
│
└── server/
    ├── app.py                 MODIFIED: lifespan wiring + middleware add
    └── config.py              MODIFIED: new fields + env parsing
```

Dependency direction is strictly one-way:

```
server/app.py → ratelimit/middleware → limiter → redis_client → [redis pkg]
server/app.py → database/factory → backends/* → [asyncpg pkg]
ratelimit ✕→✕ database  (no cross-dependency)
```

Both modules import optional packages via `importlib.import_module(variable)`
to bypass Pyright `reportMissingImports` — the established project pattern.

## 4. Invariants

Three invariants must hold across all implementation work:

1. **Import-safe without optional deps**: `import topsport_agent` succeeds
   with or without `--group db` / `--group redis` installed. Unit tests run
   green without these groups (rate-limit integration tests `pytest.skip` when
   local Redis is unavailable).
2. **Default off, zero cost**: `enable_database=False` and
   `enable_rate_limit=False` (the defaults) mean no new code path is hit; no
   middleware is installed; no connection is opened.
3. **Fail-fast on startup, fail-open at runtime** (rate limit):
   `enable_rate_limit=True` but Redis unreachable at startup → `RuntimeError`
   before the server binds. Redis going down mid-request → one log line
   + metric bump + request is allowed through (configurable via
   `RATELIMIT_FAIL_OPEN`).

## 5. Database Abstraction Layer

### 5.1 Core Protocol

```python
class Database(Protocol):
    @property
    def dialect(self) -> str: ...   # "postgres" | "mysql" | "sqlite" | "null"

    async def connect(self) -> None: ...       # build pool; idempotent
    async def close(self) -> None: ...         # tear down; idempotent
    async def health_check(self) -> bool: ...  # SELECT 1

    async def execute(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> int: ...                               # returns affected rowcount

    async def fetch_one(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any] | None: ...

    async def fetch_all(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> list[Mapping[str, Any]]: ...

    async def fetch_val(
        self, sql: str, params: Mapping[str, Any] | None = None
    ) -> Any: ...                               # first column of first row

    def transaction(self) -> AsyncContextManager["Transaction"]: ...


class Transaction(Protocol):
    """Same query API as Database, scoped to one transaction.
    Exceptions propagate out of the context manager → automatic rollback.
    No explicit begin/commit/rollback — the context manager owns lifecycle.
    """
    async def execute(...) -> int: ...
    async def fetch_one(...) -> Mapping[str, Any] | None: ...
    async def fetch_all(...) -> list[Mapping[str, Any]]: ...
    async def fetch_val(...) -> Any: ...
```

### 5.2 Design Decisions

**Named parameters (`:tenant_id`) are canonical at the Protocol level.** Each
backend translates to its driver-native placeholder (`$1` for Postgres, `%s`
for MySQL, `?` for SQLite). This mirrors SQLAlchemy's behavior and lets store
implementations be backend-agnostic. The trade-off: backends do one extra
string pass; not measurable in practice.

**Row type is `Mapping[str, Any]`, not dataclass.** Stores own their schema
and destructure rows themselves. This keeps the Gateway schema-free and
compatible with asyncpg `Record`, aiosqlite `Row`, and dict-based fakes.

**Transaction API is context-manager-only**. No `rollback()` / `commit()`
methods. Exception raises → driver rolls back. Success → commits on
`__aexit__`. Prevents the classic "forgot to rollback on error" bug.

### 5.3 Backends

| Backend | `connect` / `close` / `health_check` | `execute` / `fetch_*` | `transaction` |
|---|---|---|---|
| `NullGateway` | no-op; `health_check` returns `True` | `raise RuntimeError("database disabled")` | same |
| `PostgresGateway` | **Implemented**: `asyncpg.create_pool`, `pool.close`, `SELECT 1` | `raise NotImplementedError` | `raise NotImplementedError` |
| `MySQLGateway` | `raise NotImplementedError("backend not implemented yet")` | same | same |
| `SqliteGateway` | same as MySQL (slot reserved for `aiosqlite`) | same | same |

The Postgres skeleton does pool lifecycle deliberately — this lets
`server/app.py` wire DB lifespan and `/health` can observe "configured but
queries fail" clearly. Any store accidentally calling `db.execute()` before
its SQL is written crashes immediately, which is correct.

### 5.4 Errors

```python
class DatabaseError(Exception): ...
class ConnectionError(DatabaseError): ...     # pool/connect failure
class QueryError(DatabaseError): ...          # SQL/syntax error
class IntegrityError(QueryError): ...         # unique / FK violation
class TransactionError(DatabaseError): ...
```

Backends translate driver-native exceptions (`asyncpg.UniqueViolationError`,
`psycopg.errors.IntegrityError`, etc.) to this stable hierarchy. Stores only
catch these five classes.

### 5.5 Config & Factory

```python
@dataclass(frozen=True)
class DatabaseConfig:
    backend: str = "null"           # "null" | "postgres" | "mysql" | "sqlite"
    url: str | None = None          # dsn
    pool_min: int = 1
    pool_max: int = 10
    timeout_seconds: float = 30.0

def create_database(config: DatabaseConfig) -> Database:
    if config.backend == "null":
        return NullGateway()
    if config.backend == "postgres":
        return PostgresGateway(config)     # raises ImportError if asyncpg missing
    if config.backend in ("mysql", "sqlite"):
        raise NotImplementedError(f"{config.backend} backend not implemented yet")
    raise ValueError(f"unknown backend: {config.backend}")
```

When `enable_database=True` and `database_backend` is unset, the default is
`"postgres"` (matches the spec's "Postgres first" intent). When
`enable_database=False`, `NullGateway()` is used regardless of backend
setting.

## 6. Rate Limiting Subsystem

### 6.1 Types

```python
class RateLimitScope(str, Enum):
    IP = "ip"
    PRINCIPAL = "principal"
    TENANT = "tenant"
    ROUTE = "route"

@dataclass(frozen=True)
class RateLimitRule:
    scope: RateLimitScope
    identity: str           # "1.2.3.4" | "user-42" | "acme" | "POST:/chat"
    limit: int              # requests allowed in window
    window_seconds: int

@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    denied_scope: RateLimitScope | None
    limit: int              # limit of the tightest (or denying) rule
    remaining: int
    reset_at_ms: int
    retry_after_seconds: int
```

### 6.2 RateLimiter Protocol

```python
class RateLimiter(Protocol):
    async def check(self, rules: Sequence[RateLimitRule]) -> RateLimitDecision:
        """Atomic multi-dimension check.
        - If ANY rule is over quota, deny and do NOT increment any counter.
        - If ALL rules pass, record the request in every dimension.
        """
```

Atomicity is non-negotiable: without it, you get "IP rule passes, tenant rule
denied → IP counter already incremented → orphaned counter" on every denial.
The implementation uses a single Lua script for all rules.

### 6.3 Sliding Window Lua Script

File: `ratelimit/lua/sliding_window.lua`, version-pinned in its comment header.

```lua
-- version: 1
-- KEYS[i]: ZSET key per rule ("ratelimit:sw:v1:{scope}:{identity}")
-- ARGV[1]: now_ms
-- ARGV[2]: unique member suffix (uuid fragment, avoids ZADD dedup)
-- ARGV[3..]: limit_i, window_ms_i  (alternating pairs)
local now = tonumber(ARGV[1])
local suffix = ARGV[2]
local n = #KEYS

-- Phase 1: check all, short-circuit on first over-quota
for i = 1, n do
  local limit = tonumber(ARGV[1 + i*2])
  local window = tonumber(ARGV[2 + i*2])
  redis.call('ZREMRANGEBYSCORE', KEYS[i], 0, now - window)
  local count = redis.call('ZCARD', KEYS[i])
  if count >= limit then
    return {0, i, count, limit, now + window}
  end
end

-- Phase 2: all passed, record in each dimension
for i = 1, n do
  local window = tonumber(ARGV[2 + i*2])
  redis.call('ZADD', KEYS[i], now, now .. ':' .. suffix)
  redis.call('PEXPIRE', KEYS[i], window)
end
return {1, 0, 0, 0, 0}
```

Runtime flow:

1. On server startup: `SCRIPT LOAD` → cache sha1.
2. Per request: `EVALSHA sha keys args`; on `NOSCRIPT` error reload and retry.
3. Every ZSET gets `PEXPIRE` so a key nobody hits again expires automatically
   — prevents unbounded memory growth even if `ZREMRANGEBYSCORE` misses.

### 6.4 FastAPI Middleware

```python
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # 1. exempt paths (/health, /metrics, /docs, /openapi.json, /redoc)
        # 2. extract identity: ip, principal, tenant, route-template
        # 3. build rules from config (skip rules with limit == 0)
        # 4. try limiter.check(rules)
        #    - on exception: bump ratelimit_degraded_total; fail-open or 503
        # 5. if denied: 429 with headers (Retry-After, X-RateLimit-*)
        # 6. else: call_next(request) + attach X-RateLimit-* headers
```

**Critical wiring details** (documented in code comments):

- **Route identity uses the template, not the raw path.** `/chat/123` and
  `/chat/456` share the `/chat/{id}` template; otherwise path-parameterized
  routes blow up the key space. Pull `request.scope["route"].path`.
- **X-Forwarded-For handling** is gated on `ratelimit_trust_forwarded_for`.
  Default `False`. When `True`, take the leftmost hop (first client in the
  chain). When `False`, `request.client.host` (direct peer).
- **Rate-limit middleware must run AFTER auth middleware** (so
  `request.state.principal` / `tenant_id` are populated before the limiter
  looks at them). FastAPI executes middlewares in **reverse order of
  `add_middleware` calls** (last added = first to run). Therefore the *code*
  order is: call `app.add_middleware(RateLimitMiddleware, ...)` **first**,
  then `app.add_middleware(AuthMiddleware, ...)`. Counter-intuitive but
  correct — add a comment at the add site.

### 6.5 Metrics (Prometheus)

Optional, gated on `metrics` dependency group. Reuses the pattern in
`engine/permission/metrics.py`:

- `ratelimit_requests_total{scope}` — counter
- `ratelimit_denied_total{scope}` — counter
- `ratelimit_degraded_total{reason}` — counter (Redis errors during check)
- `ratelimit_check_duration_seconds` — histogram

Without the `metrics` group, all counters are no-op stubs.

## 7. Server Configuration

Added to `ServerConfig` (`src/topsport_agent/server/config.py`):

```python
# Database
enable_database: bool = False
database_backend: str = "postgres"       # applied only when enable_database=True
database_url: str | None = None
database_pool_min: int = 1
database_pool_max: int = 10
database_timeout_seconds: float = 30.0

# Rate limit
enable_rate_limit: bool = False
ratelimit_redis_url: str | None = None
ratelimit_window_seconds: int = 60
ratelimit_per_ip: int = 300              # 0 = disable this dimension
ratelimit_per_principal: int = 60
ratelimit_per_tenant: int = 1000
ratelimit_per_route_default: int = 0
ratelimit_routes: dict[str, int] = field(default_factory=dict)
ratelimit_exempt_paths: frozenset[str] = frozenset({
    "/health", "/metrics", "/docs", "/openapi.json", "/redoc",
})
ratelimit_trust_forwarded_for: bool = False
ratelimit_fail_open: bool = True
```

### 7.1 Environment Variable Binding

| env | field | default |
|---|---|---|
| `ENABLE_DATABASE` | `enable_database` | `false` |
| `DATABASE_BACKEND` | `database_backend` | `postgres` |
| `DATABASE_URL` | `database_url` | `None` |
| `DATABASE_POOL_MIN` / `DATABASE_POOL_MAX` | pool sizes | `1` / `10` |
| `ENABLE_RATE_LIMIT` | `enable_rate_limit` | `false` |
| `RATELIMIT_REDIS_URL` | `ratelimit_redis_url` | `None` |
| `RATELIMIT_PER_IP` / `_PRINCIPAL` / `_TENANT` | limits | `300` / `60` / `1000` |
| `RATELIMIT_WINDOW_SECONDS` | window | `60` |
| `RATELIMIT_ROUTES` | JSON `{"/chat":20}` | `{}` |
| `RATELIMIT_EXEMPT_PATHS` | CSV `"/foo,/bar"`, additive | — |
| `RATELIMIT_TRUST_FORWARDED_FOR` | enable XFF | `false` |
| `RATELIMIT_FAIL_OPEN` | runtime Redis failure policy | `true` |

### 7.2 Lifespan Wiring

```python
async def lifespan(app: FastAPI):
    cfg = app.state.config

    # Database
    if cfg.enable_database:
        db = create_database(DatabaseConfig.from_server(cfg))
        await db.connect()
        if not await db.health_check():
            raise RuntimeError("database enabled but health_check failed")
        app.state.database = db
    else:
        app.state.database = NullGateway()

    # Rate limiting
    if cfg.enable_rate_limit:
        if not cfg.ratelimit_redis_url:
            raise RuntimeError("ENABLE_RATE_LIMIT=true but RATELIMIT_REDIS_URL unset")
        client = create_redis_client(cfg.ratelimit_redis_url)
        if not await client.ping():
            raise RuntimeError("rate limit enabled but Redis ping failed")
        sha = await client.script_load(LUA_SCRIPT)
        limiter = RedisSlidingWindowLimiter(client=client, sha=sha, script=LUA_SCRIPT)
        app.state.ratelimit_client = client
        app.state.ratelimit_limiter = limiter

    yield

    if cfg.enable_rate_limit:
        await app.state.ratelimit_client.close()
    if cfg.enable_database and not isinstance(app.state.database, NullGateway):
        await app.state.database.close()
```

Middleware is installed unconditionally at module load time but reads
`app.state.ratelimit_limiter` per request; when rate limit is disabled it
short-circuits to `call_next` immediately.

## 8. Testing Strategy

Six new test files; all pass with/without optional dependency groups.

| Test file | Scope | Dependencies |
|---|---|---|
| `test_database_gateway.py` | `NullGateway` behavior + Protocol conformance | none |
| `test_database_factory.py` | Backend selection, missing-dep error, invalid backend | none |
| `test_database_backends_postgres.py` | `PostgresGateway` pool lifecycle + `health_check` + `execute` raising `NotImplementedError` | `--group db` (skip if asyncpg missing); does NOT require a running PG (uses asyncpg's `create_pool` failure paths) |
| `test_ratelimit_limiter.py` | Sliding window semantics, multi-dimension atomicity, concurrency | **local Redis** (skip if `ping` fails) |
| `test_ratelimit_middleware.py` | End-to-end via `TestClient`: 429, headers, exempt paths, fail-open | **local Redis** |
| `test_server_lifespan_integration.py` | enable=False path (no Redis call), enable=True + bad URL fails fast | none |

### 8.1 Local Redis Fixture

```python
@pytest.fixture
async def redis_client():
    url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
    try:
        client = create_redis_client(url)
        if not await client.ping():
            pytest.skip("local Redis not available")
    except Exception:
        pytest.skip("local Redis not available")
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.close()
```

Testing invariants:

1. **Opt-in local Redis**: if Redis unavailable → skip, not fail. Developers
   without Redis installed still see green `pytest`.
2. **DB 15 by default** — isolated from developers' own Redis scratch space.
3. **FLUSHDB before and after** every test — order-independent.
4. **No FakeRedis** — user-directed: integration tests run against real Redis
   to catch actual Lua / RESP behavior.

## 9. Deliverables

### 9.1 New files (24: 18 source + 6 test)

```
src/topsport_agent/database/__init__.py
src/topsport_agent/database/gateway.py
src/topsport_agent/database/config.py
src/topsport_agent/database/errors.py
src/topsport_agent/database/factory.py
src/topsport_agent/database/backends/__init__.py
src/topsport_agent/database/backends/null.py
src/topsport_agent/database/backends/postgres.py
src/topsport_agent/database/backends/mysql.py
src/topsport_agent/database/backends/sqlite.py

src/topsport_agent/ratelimit/__init__.py
src/topsport_agent/ratelimit/types.py
src/topsport_agent/ratelimit/limiter.py
src/topsport_agent/ratelimit/redis_client.py
src/topsport_agent/ratelimit/middleware.py
src/topsport_agent/ratelimit/metrics.py
src/topsport_agent/ratelimit/config.py
src/topsport_agent/ratelimit/lua/sliding_window.lua

tests/test_database_gateway.py
tests/test_database_factory.py
tests/test_database_backends_postgres.py
tests/test_ratelimit_limiter.py
tests/test_ratelimit_middleware.py
tests/test_server_lifespan_integration.py
```

### 9.2 Modified files (4)

- `src/topsport_agent/server/config.py` — add fields + env parsing
- `src/topsport_agent/server/app.py` — lifespan wiring + middleware register
- `pyproject.toml` — add `db` and `redis` dependency groups
- `README.md` — two new sections (Database, Rate Limiting)

## 10. Acceptance Criteria

### 10.1 Core

- [ ] `uv sync && uv run pytest -v` passes without `--group db` / `--group redis` (Redis integration tests skip gracefully).
- [ ] `uv sync --group db --group redis --group dev` with local Redis running → `uv run pytest -v` passes fully.
- [ ] `grep -rn "^import asyncpg\|^import redis\|^from asyncpg\|^from redis" src/topsport_agent/` returns nothing — all optional imports go through `importlib.import_module(variable)`.

### 10.2 Architecture

- [ ] `python -c "from topsport_agent.database import create_database"` succeeds without `asyncpg` installed.
- [ ] `python -c "from topsport_agent.ratelimit import RateLimitMiddleware"` succeeds without `redis` installed.
- [ ] Starting the server with `ENABLE_DATABASE=false ENABLE_RATE_LIMIT=false` never triggers any import of `asyncpg` / `redis`.

### 10.3 Rate Limiting (with real Redis)

- [ ] `RATELIMIT_PER_IP=3 WINDOW=60` → 4th rapid request returns 429 with `Retry-After`, `X-RateLimit-*` headers.
- [ ] `RATELIMIT_PER_TENANT=5` → over-quota rejects even when IP/principal change.
- [ ] `/health`, `/docs`, `/openapi.json`, `/redoc` bypass rate limiting.
- [ ] Kill Redis mid-run → requests still served (fail-open), `ratelimit_degraded_total` counter increments.
- [ ] `ENABLE_RATE_LIMIT=true` but Redis unreachable → `RuntimeError` at startup (fail-fast).

### 10.4 Database

- [ ] `ENABLE_DATABASE=false` under any request path — no `asyncpg` call anywhere.
- [ ] `ENABLE_DATABASE=true DATABASE_BACKEND=postgres` + bad URL → `RuntimeError` at startup.
- [ ] `ENABLE_DATABASE=true DATABASE_BACKEND=postgres` + good URL → `db.health_check()` returns `True`; `db.execute(...)` raises `NotImplementedError` (deliberate).
- [ ] `DATABASE_BACKEND=mysql` or `sqlite` → `NotImplementedError` from backend constructor.

## 11. Follow-up Specs (not in this one)

These are the natural next steps, each deserving its own spec:

- **Store ↔ DB wiring**: implement `PostgresAuditStore`, `PostgresPersonaRegistry`, `PostgresAssignmentStore` using the `Database` Gateway. Includes schema, migrations, real SQL.
- **Session persistence + crash recovery**: persist `SessionStore` to DB; hot cache in Redis; SSE stream resume via Last-Event-ID.
- **KillSwitch cross-replica broadcast**: move `KillSwitchGate` state to Redis pub/sub.
- **Rate limit advanced features**: token-bucket, per-tool limits, dynamic quota admin API.

## 12. References

- [redis-py async](https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html)
- [asyncpg connection pools](https://magicstack.github.io/asyncpg/current/api/index.html#connection-pools)
- [FastAPI middleware order](https://fastapi.tiangolo.com/tutorial/middleware/)
- Related learnings in `.learnings/LEARNINGS.md`:
  - "Pyright's optional-dependency escape hatch"
  - "Test-injectable factories let optional deps stay optional"
  - "Fail-closed is the only safe default for permission machinery" (contrast:
    rate limit uses fail-open because it's governance, not security)
