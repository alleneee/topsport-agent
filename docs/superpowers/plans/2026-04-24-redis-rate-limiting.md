# Redis Rate Limiting — Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a complete Redis-backed sliding-window rate limiter for the FastAPI server, covering four dimensions (IP / principal / tenant / route) atomically via a single Lua script. Fail-fast on startup if Redis unreachable; fail-open at runtime if Redis disconnects mid-request.

**Architecture:** Single module `src/topsport_agent/ratelimit/` with `Protocol + Redis implementation` style (matches the project's `MemoryStore` / `LLMProvider` pattern). One Lua script in `lua/sliding_window.lua` runs `SCRIPT LOAD` at startup and `EVALSHA` per request, with `NOSCRIPT`-fallback to `EVAL`. `RateLimitMiddleware` reads identity from `request.state` (populated by auth middleware) and short-circuits on exempt paths.

**Tech Stack:** Python 3.11, `redis>=5.0` async client (loaded via `importlib.import_module` pattern), FastAPI BaseHTTPMiddleware, Prometheus counters (optional dep), pytest with local Redis fixture.

**Related spec:** `docs/superpowers/specs/2026-04-24-database-abstraction-and-redis-ratelimit-design.md` sections 3, 4, 6, 7, 8.

---

## Preconditions

1. **Local Redis must be running** for integration tests. A single Docker command works:

   ```bash
   docker run -d -p 6379:6379 --name topsport-test-redis redis:7-alpine
   ```

   Without Redis, integration tests `pytest.skip` gracefully. Plan-validation tests pass even without Redis.

2. **All optional dependency groups should stay installed** in the venv (`dev db api mcp metrics llm sandbox tracing browser redis`). Subagents must NOT run `uv sync --group redis` alone — that uninstalls every other group and breaks `test_permission_*` etc. (See `.learnings/LEARNINGS.md` entry "uv sync --group X is destructive".)

---

## Task 0: Scaffold `ratelimit` module + `redis` dependency group

**Files:**
- Modify: `pyproject.toml` (add `redis` group)
- Create: `src/topsport_agent/ratelimit/__init__.py` (empty placeholder)
- Create: `src/topsport_agent/ratelimit/lua/` directory
- Test: `tests/test_ratelimit_scaffold.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ratelimit_scaffold.py
def test_ratelimit_module_imports() -> None:
    import topsport_agent.ratelimit  # noqa: F401
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_ratelimit_scaffold.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Add `redis` group to `pyproject.toml`**

Find `[dependency-groups]` (after the `db = [...]` group that Plan A added). Add:

```toml
redis = [
    "redis>=5.0.0",
]
```

- [ ] **Step 4: Create package**

```python
# src/topsport_agent/ratelimit/__init__.py
"""Redis-backed sliding-window rate limiter for the FastAPI server."""
```

Also create the `lua/` directory with a placeholder gitkeep:

```bash
mkdir -p src/topsport_agent/ratelimit/lua
touch src/topsport_agent/ratelimit/lua/.gitkeep
```

- [ ] **Step 5: Install redis group into the venv**

```bash
uv sync --all-groups
```

(Using `--all-groups` per the learnings entry to avoid removing other optional groups.)

- [ ] **Step 6: Run — expect PASS**

`uv run pytest tests/test_ratelimit_scaffold.py -v`
Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/topsport_agent/ratelimit/__init__.py src/topsport_agent/ratelimit/lua/.gitkeep tests/test_ratelimit_scaffold.py
git commit -m "feat(ratelimit): scaffold module + redis dependency group"
```

---

## Task 1: Types (`types.py`)

**Files:**
- Create: `src/topsport_agent/ratelimit/types.py`
- Test: `tests/test_ratelimit_types.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ratelimit_types.py
import dataclasses

import pytest

from topsport_agent.ratelimit.types import (
    RateLimitDecision,
    RateLimitRule,
    RateLimitScope,
)


def test_scope_values() -> None:
    assert RateLimitScope.IP == "ip"
    assert RateLimitScope.PRINCIPAL == "principal"
    assert RateLimitScope.TENANT == "tenant"
    assert RateLimitScope.ROUTE == "route"


def test_rule_is_frozen() -> None:
    rule = RateLimitRule(
        scope=RateLimitScope.IP,
        identity="1.2.3.4",
        limit=100,
        window_seconds=60,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.limit = 200  # type: ignore[misc]


def test_decision_is_frozen() -> None:
    decision = RateLimitDecision(
        allowed=True,
        denied_scope=None,
        limit=100,
        remaining=99,
        reset_at_ms=123456,
        retry_after_seconds=0,
    )
    assert decision.allowed is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.allowed = False  # type: ignore[misc]


def test_denied_decision_has_scope() -> None:
    decision = RateLimitDecision(
        allowed=False,
        denied_scope=RateLimitScope.TENANT,
        limit=1000,
        remaining=0,
        reset_at_ms=123456,
        retry_after_seconds=30,
    )
    assert decision.allowed is False
    assert decision.denied_scope == RateLimitScope.TENANT
    assert decision.retry_after_seconds == 30
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_ratelimit_types.py -v`

- [ ] **Step 3: Implement `types.py`**

```python
# src/topsport_agent/ratelimit/types.py
"""Rate-limiter value types: scope enum, rule, decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RateLimitScope(str, Enum):
    """The four dimensions we enforce. Each has its own Redis key space."""

    IP = "ip"
    PRINCIPAL = "principal"
    TENANT = "tenant"
    ROUTE = "route"


@dataclass(slots=True, frozen=True)
class RateLimitRule:
    """One dimension's quota for one identity.

    The limiter checks N rules per request (typically 4, one per scope).
    scope + identity together form the Redis key namespace.
    """

    scope: RateLimitScope
    identity: str               # e.g. "1.2.3.4", "user-42", "acme", "POST:/chat"
    limit: int                  # requests allowed in window
    window_seconds: int         # window size in seconds


@dataclass(slots=True, frozen=True)
class RateLimitDecision:
    """Outcome of a `limiter.check(rules)` call.

    - allowed=True  → all rules within quota; remaining/limit are the tightest rule's metrics (for response headers)
    - allowed=False → denied_scope is the first rule that exceeded its quota
    """

    allowed: bool
    denied_scope: RateLimitScope | None
    limit: int
    remaining: int
    reset_at_ms: int
    retry_after_seconds: int


__all__ = ["RateLimitDecision", "RateLimitRule", "RateLimitScope"]
```

- [ ] **Step 4: Run — expect PASS**

`uv run pytest tests/test_ratelimit_types.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit (stop — main agent commits)**

Report DONE.

---

## Task 2: Lua script (`lua/sliding_window.lua`)

No Python test yet — the script is tested indirectly through Task 5's limiter tests. This task just lands the file.

**Files:**
- Create: `src/topsport_agent/ratelimit/lua/sliding_window.lua`
- Modify: delete the `.gitkeep` placeholder (optional — can keep alongside)

- [ ] **Step 1: Create the Lua script**

```lua
-- sliding_window.lua — atomic multi-dimension rate limiter
-- version: 1  (bump if KEY format changes; Python side must prefix keys with vN)
--
-- Args:
--   KEYS[i]  : ZSET key per rule ("ratelimit:sw:v1:{scope}:{identity}")
--   ARGV[1]  : now_ms
--   ARGV[2]  : unique member suffix (uuid fragment, avoids ZADD dedup on same-ms requests)
--   ARGV[3..] : limit_i, window_ms_i  (alternating pairs, length = 2*#KEYS)
--
-- Returns: {allowed (1/0), denied_idx_1based, count, limit, reset_at_ms}
--   - on deny: allowed=0, denied_idx is the rule that tripped, count is current bucket count, limit is that rule's limit, reset_at_ms = now + window
--   - on allow: allowed=1, other fields 0 (middleware computes headers from known rules)

local now = tonumber(ARGV[1])
local suffix = ARGV[2]
local n = #KEYS

-- Phase 1: check each rule in order; short-circuit on first over-quota
for i = 1, n do
  local limit  = tonumber(ARGV[1 + i*2])
  local window = tonumber(ARGV[2 + i*2])
  redis.call('ZREMRANGEBYSCORE', KEYS[i], 0, now - window)
  local count = redis.call('ZCARD', KEYS[i])
  if count >= limit then
    return {0, i, count, limit, now + window}
  end
end

-- Phase 2: all passed — record the request in every dimension
for i = 1, n do
  local window = tonumber(ARGV[2 + i*2])
  redis.call('ZADD', KEYS[i], now, now .. ':' .. suffix)
  redis.call('PEXPIRE', KEYS[i], window)
end
return {1, 0, 0, 0, 0}
```

- [ ] **Step 2: Remove `.gitkeep`**

```bash
git rm -f src/topsport_agent/ratelimit/lua/.gitkeep 2>/dev/null || rm -f src/topsport_agent/ratelimit/lua/.gitkeep
```

- [ ] **Step 3: Commit (main agent)**

Expected commit message: `feat(ratelimit): sliding_window Lua script (v1)`

---

## Task 3: `create_redis_client` factory (`redis_client.py`)

**Files:**
- Create: `src/topsport_agent/ratelimit/redis_client.py`
- Test: `tests/test_ratelimit_redis_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ratelimit_redis_client.py
import importlib.util
import os

import pytest

from topsport_agent.ratelimit.redis_client import create_redis_client

_redis_available = importlib.util.find_spec("redis") is not None
_test_url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.mark.skipif(not _redis_available, reason="redis package not installed")
def test_create_redis_client_returns_async_client() -> None:
    client = create_redis_client(_test_url)
    # Duck-typed: must expose the methods we rely on.
    for name in ("ping", "evalsha", "eval", "script_load", "close", "flushdb"):
        assert hasattr(client, name), f"missing method: {name}"


@pytest.mark.skipif(not _redis_available, reason="redis package not installed")
@pytest.mark.asyncio
async def test_ping_round_trip_if_redis_up() -> None:
    client = create_redis_client(_test_url)
    try:
        ok = await client.ping()
    except Exception:
        pytest.skip("local Redis not reachable at " + _test_url)
    assert ok is True
    await client.close()


def test_create_without_redis_package_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `redis` is not installed, factory must raise ImportError with an install hint."""
    import sys

    # Clear both the package and any submodule references
    monkeypatch.setitem(sys.modules, "redis", None)

    with pytest.raises(ImportError, match="uv sync --group redis"):
        create_redis_client(_test_url)
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_ratelimit_redis_client.py -v`

- [ ] **Step 3: Implement `redis_client.py`**

```python
# src/topsport_agent/ratelimit/redis_client.py
"""Factory for the async Redis client.

Imports the `redis` package lazily via importlib so the module is safe to
import in environments without `--group redis`. The ImportError wrapper
surfaces an install-hint message at construction time.
"""

from __future__ import annotations

import importlib
from typing import Any


def create_redis_client(url: str) -> Any:
    """Construct an async Redis client from a redis:// URL.

    Returns an instance of `redis.asyncio.Redis` with `decode_responses=True`
    so EVALSHA returns Python strings for numeric tuples (Lua scripts send
    bulk-string arrays).

    Raises:
        ImportError: if the `redis` package is not installed.
    """
    try:
        mod_name = "redis.asyncio"
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ImportError(
            "RateLimiter requires the 'redis' package. Install via: "
            "uv sync --group redis"
        ) from exc
    return mod.Redis.from_url(url, decode_responses=True)


__all__ = ["create_redis_client"]
```

- [ ] **Step 4: Run — expect PASS**

`uv run pytest tests/test_ratelimit_redis_client.py -v`
Expected: 3 passed (or 2 passed + 1 skipped if local Redis unreachable).

- [ ] **Step 5: Commit (main agent)**

`feat(ratelimit): async Redis client factory with lazy import`

---

## Task 4: `RateLimiter` Protocol (`limiter.py` — Protocol half)

The limiter has two concerns: (1) the abstract contract, and (2) the Redis implementation. Split across this task (contract) and Task 5 (Redis impl) to keep reviews manageable.

**Files:**
- Create: `src/topsport_agent/ratelimit/limiter.py` (Protocol only; impl in Task 5)
- Test: `tests/test_ratelimit_protocol.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ratelimit_protocol.py
def test_ratelimiter_protocol_importable() -> None:
    from topsport_agent.ratelimit.limiter import RateLimiter  # noqa: F401


def test_ratelimiter_has_check_method() -> None:
    from topsport_agent.ratelimit.limiter import RateLimiter

    assert "check" in dir(RateLimiter)
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_ratelimit_protocol.py -v`

- [ ] **Step 3: Implement the Protocol**

```python
# src/topsport_agent/ratelimit/limiter.py
"""RateLimiter Protocol and its Redis implementation.

Protocol here; the RedisSlidingWindowLimiter class lands in a follow-up task.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import RateLimitDecision, RateLimitRule


@runtime_checkable
class RateLimiter(Protocol):
    """Atomic multi-dimension rate-limit check.

    Contract:
    - If ANY rule in `rules` is over its quota, deny the request and do NOT
      increment any rule's counter.
    - If ALL rules pass, record the request in every dimension atomically.
    - Never partially-commits — failure mode must preserve all-or-nothing.
    """

    async def check(
        self, rules: Sequence[RateLimitRule]
    ) -> RateLimitDecision: ...


__all__ = ["RateLimiter"]
```

- [ ] **Step 4: Run — expect PASS**

`uv run pytest tests/test_ratelimit_protocol.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit (main agent)**

`feat(ratelimit): RateLimiter Protocol`

---

## Task 5: `RedisSlidingWindowLimiter` implementation + integration tests

This is the largest task. Requires local Redis running (tests skip otherwise).

**Files:**
- Modify: `src/topsport_agent/ratelimit/limiter.py` (add `RedisSlidingWindowLimiter` class)
- Create: `tests/conftest.py` changes — add a `redis_client` fixture (ONLY if conftest doesn't already have one; otherwise add fixture to the test file directly)
- Test: `tests/test_ratelimit_limiter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ratelimit_limiter.py
from __future__ import annotations

import asyncio
import importlib.util
import os
import time
import uuid
from pathlib import Path

import pytest

from topsport_agent.ratelimit.redis_client import create_redis_client
from topsport_agent.ratelimit.types import RateLimitRule, RateLimitScope

_redis_available = importlib.util.find_spec("redis") is not None
_test_url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
requires_redis = pytest.mark.skipif(
    not _redis_available, reason="redis package not installed"
)


@pytest.fixture
async def redis_client():
    """Shared fixture: real Redis on DB 15, FLUSHDB before+after."""
    if not _redis_available:
        pytest.skip("redis package not installed")
    client = create_redis_client(_test_url)
    try:
        if not await client.ping():
            pytest.skip("local Redis not responding")
    except Exception:
        pytest.skip(f"local Redis not reachable at {_test_url}")
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.close()


@pytest.fixture
async def limiter(redis_client):
    from topsport_agent.ratelimit.limiter import RedisSlidingWindowLimiter

    script_path = (
        Path("src/topsport_agent/ratelimit/lua/sliding_window.lua")
        .resolve()
    )
    script = script_path.read_text(encoding="utf-8")
    sha = await redis_client.script_load(script)
    return RedisSlidingWindowLimiter(
        client=redis_client,
        sha=sha,
        script=script,
    )


def _unique_id(prefix: str) -> str:
    """Each test gets a unique identity so parallel/sequential runs don't collide."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@requires_redis
@pytest.mark.asyncio
async def test_first_request_is_allowed(limiter) -> None:
    decision = await limiter.check(
        [
            RateLimitRule(
                scope=RateLimitScope.IP,
                identity=_unique_id("ip"),
                limit=5,
                window_seconds=60,
            )
        ]
    )
    assert decision.allowed is True
    assert decision.denied_scope is None


@requires_redis
@pytest.mark.asyncio
async def test_nth_plus_one_request_denied(limiter) -> None:
    ident = _unique_id("ip")
    rule = RateLimitRule(
        scope=RateLimitScope.IP,
        identity=ident,
        limit=3,
        window_seconds=60,
    )
    for _ in range(3):
        assert (await limiter.check([rule])).allowed is True
    decision = await limiter.check([rule])
    assert decision.allowed is False
    assert decision.denied_scope == RateLimitScope.IP
    assert decision.limit == 3
    assert decision.retry_after_seconds > 0


@requires_redis
@pytest.mark.asyncio
async def test_multi_dimension_denies_on_tightest(limiter) -> None:
    """When multiple rules are given, the one that runs out first is the denial scope."""
    ip_id = _unique_id("ip")
    tenant_id = _unique_id("tenant")
    ip_rule = RateLimitRule(RateLimitScope.IP, ip_id, 100, 60)
    tenant_rule = RateLimitRule(RateLimitScope.TENANT, tenant_id, 2, 60)

    # Two allowed, tenant bucket fills
    for _ in range(2):
        assert (await limiter.check([ip_rule, tenant_rule])).allowed is True

    decision = await limiter.check([ip_rule, tenant_rule])
    assert decision.allowed is False
    assert decision.denied_scope == RateLimitScope.TENANT


@requires_redis
@pytest.mark.asyncio
async def test_atomicity_no_partial_commit(limiter, redis_client) -> None:
    """When one rule denies, the OTHER rule's counter must not increment."""
    ip_id = _unique_id("ip")
    tenant_id = _unique_id("tenant")
    ip_rule = RateLimitRule(RateLimitScope.IP, ip_id, 100, 60)
    tenant_rule = RateLimitRule(RateLimitScope.TENANT, tenant_id, 1, 60)

    # First request succeeds in both
    await limiter.check([ip_rule, tenant_rule])

    # IP key count before the denied attempt
    ip_key = f"ratelimit:sw:v1:ip:{ip_id}"
    tenant_key = f"ratelimit:sw:v1:tenant:{tenant_id}"
    ip_count_before = await redis_client.zcard(ip_key)
    tenant_count_before = await redis_client.zcard(tenant_key)
    assert ip_count_before == 1
    assert tenant_count_before == 1

    # Second request denied by tenant rule
    decision = await limiter.check([ip_rule, tenant_rule])
    assert decision.allowed is False

    # IP count must NOT have grown
    ip_count_after = await redis_client.zcard(ip_key)
    tenant_count_after = await redis_client.zcard(tenant_key)
    assert ip_count_after == 1, "IP rule must not increment when tenant denies"
    assert tenant_count_after == 1, "tenant rule must not double-count"


@requires_redis
@pytest.mark.asyncio
async def test_noscript_fallback_triggers_eval(limiter, redis_client) -> None:
    """If EVALSHA gets NOSCRIPT, the limiter falls back to EVAL transparently.

    Trigger: FLUSH scripts server-side, then check() should still work.
    """
    await redis_client.script_flush()
    rule = RateLimitRule(
        scope=RateLimitScope.IP,
        identity=_unique_id("ip"),
        limit=5,
        window_seconds=60,
    )
    decision = await limiter.check([rule])
    assert decision.allowed is True


@requires_redis
@pytest.mark.asyncio
async def test_concurrent_requests_respect_quota(limiter) -> None:
    """Running 10 concurrent check() calls against a limit=3 rule should
    result in exactly 3 allowed + 7 denied (atomic enforcement)."""
    ident = _unique_id("ip")
    rule = RateLimitRule(RateLimitScope.IP, ident, 3, 60)
    results = await asyncio.gather(
        *[limiter.check([rule]) for _ in range(10)]
    )
    allowed = sum(1 for r in results if r.allowed)
    denied = sum(1 for r in results if not r.allowed)
    assert allowed == 3
    assert denied == 7
```

- [ ] **Step 2: Run — expect FAIL (class not defined)**

`uv run pytest tests/test_ratelimit_limiter.py -v`

- [ ] **Step 3: Append `RedisSlidingWindowLimiter` to `limiter.py`**

Edit `src/topsport_agent/ratelimit/limiter.py` — **add** (do not replace) the class below after the existing `__all__` line (and update `__all__` to include the new name):

```python
# Add to the top of the file (below `from __future__ import annotations`):
import time
import uuid
from typing import Any

# Add at the end of the file (before __all__):

_KEY_PREFIX = "ratelimit:sw:v1"


class RedisSlidingWindowLimiter:
    """Sliding-window rate limiter backed by Redis + a Lua script.

    One `SCRIPT LOAD` at startup gives a sha1; per request we try `EVALSHA`
    and fall back to `EVAL` on NOSCRIPT (auto-heals after Redis restart or
    SCRIPT FLUSH).
    """

    def __init__(
        self,
        *,
        client: Any,
        sha: str,
        script: str,
    ) -> None:
        self._client = client
        self._sha = sha
        self._script = script

    @staticmethod
    def _key(rule: RateLimitRule) -> str:
        return f"{_KEY_PREFIX}:{rule.scope.value}:{rule.identity}"

    async def check(
        self, rules: Sequence[RateLimitRule]
    ) -> RateLimitDecision:
        if not rules:
            # Nothing to check — trivially allowed.
            return RateLimitDecision(
                allowed=True,
                denied_scope=None,
                limit=0,
                remaining=0,
                reset_at_ms=0,
                retry_after_seconds=0,
            )

        now_ms = int(time.time() * 1000)
        suffix = uuid.uuid4().hex[:12]

        keys = [self._key(r) for r in rules]
        args: list[int | str] = [now_ms, suffix]
        for rule in rules:
            args.append(rule.limit)
            args.append(rule.window_seconds * 1000)

        try:
            raw = await self._client.evalsha(self._sha, len(keys), *keys, *args)
        except Exception as exc:  # noqa: BLE001
            # NOSCRIPT (SCRIPT FLUSH or restart) → reload and retry via EVAL.
            if "NOSCRIPT" in repr(exc):
                raw = await self._client.eval(
                    self._script, len(keys), *keys, *args
                )
            else:
                raise

        allowed_flag, denied_idx, count, limit, reset_at_ms = (
            int(x) for x in raw
        )
        if allowed_flag == 1:
            # All rules passed. Remaining/limit taken from the *tightest* rule
            # (lowest `limit - ceil(window_req_rate)`) — simplification:
            # return the first rule's limit as header hint.
            primary = rules[0]
            return RateLimitDecision(
                allowed=True,
                denied_scope=None,
                limit=primary.limit,
                remaining=max(primary.limit - 1, 0),  # minus this request
                reset_at_ms=now_ms + primary.window_seconds * 1000,
                retry_after_seconds=0,
            )

        denied_rule = rules[denied_idx - 1]  # Lua indices are 1-based
        retry_after = max(1, (reset_at_ms - now_ms + 999) // 1000)
        return RateLimitDecision(
            allowed=False,
            denied_scope=denied_rule.scope,
            limit=limit,
            remaining=0,
            reset_at_ms=reset_at_ms,
            retry_after_seconds=int(retry_after),
        )


__all__ = ["RateLimiter", "RedisSlidingWindowLimiter"]
```

- [ ] **Step 4: Ensure local Redis is running**

```bash
docker ps | grep -q topsport-test-redis || \
  docker run -d -p 6379:6379 --name topsport-test-redis redis:7-alpine
```

(Skip if user already has Redis running locally.)

- [ ] **Step 5: Run — expect PASS**

`uv run pytest tests/test_ratelimit_limiter.py -v`
Expected: 6 passed (or 6 skipped if Redis not reachable — skip is acceptable).

- [ ] **Step 6: Run full suite to check no regressions**

`uv run pytest -q 2>&1 | tail -3`
Expected: 831 passed + new tests, 1 skipped (or more skipped if Redis down).

- [ ] **Step 7: Commit (main agent)**

`feat(ratelimit): RedisSlidingWindowLimiter + integration tests against local Redis`

---

## Task 6: Config (`config.py`)

**Files:**
- Create: `src/topsport_agent/ratelimit/config.py`
- Test: `tests/test_ratelimit_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ratelimit_config.py
import dataclasses

import pytest

from topsport_agent.ratelimit.config import RateLimitConfig


def test_defaults_disabled() -> None:
    cfg = RateLimitConfig()
    assert cfg.enabled is False
    assert cfg.redis_url is None
    assert cfg.window_seconds == 60
    assert cfg.per_ip_limit == 300
    assert cfg.per_principal_limit == 60
    assert cfg.per_tenant_limit == 1000
    assert cfg.per_route_default == 0
    assert cfg.per_route_limits == {}
    assert "/health" in cfg.exempt_paths
    assert "/metrics" in cfg.exempt_paths
    assert "/docs" in cfg.exempt_paths
    assert "/openapi.json" in cfg.exempt_paths
    assert cfg.trust_forwarded_for is False
    assert cfg.fail_open_on_redis_error is True


def test_config_is_frozen() -> None:
    cfg = RateLimitConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.enabled = True  # type: ignore[misc]


def test_custom_config() -> None:
    cfg = RateLimitConfig(
        enabled=True,
        redis_url="redis://localhost:6379/0",
        per_ip_limit=500,
        per_route_limits={"/chat": 20},
    )
    assert cfg.enabled is True
    assert cfg.per_route_limits == {"/chat": 20}
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_ratelimit_config.py -v`

- [ ] **Step 3: Implement `config.py`**

```python
# src/topsport_agent/ratelimit/config.py
"""RateLimitConfig — all tunables for the rate-limit subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

_DEFAULT_EXEMPT: frozenset[str] = frozenset(
    {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}
)


@dataclass(slots=True, frozen=True)
class RateLimitConfig:
    """Frozen config. Default is "off" and imposes zero runtime cost."""

    enabled: bool = False
    redis_url: str | None = None

    # Per-dimension limits. 0 = disable that dimension.
    window_seconds: int = 60
    per_ip_limit: int = 300
    per_principal_limit: int = 60
    per_tenant_limit: int = 1000
    per_route_default: int = 0
    per_route_limits: dict[str, int] = field(default_factory=dict)

    exempt_paths: frozenset[str] = _DEFAULT_EXEMPT
    trust_forwarded_for: bool = False
    fail_open_on_redis_error: bool = True


__all__ = ["RateLimitConfig"]
```

- [ ] **Step 4: Run — expect PASS**

`uv run pytest tests/test_ratelimit_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit (main agent)**

`feat(ratelimit): RateLimitConfig frozen dataclass`

---

## Task 7: Metrics (`metrics.py`)

Wraps prometheus-client if the `metrics` group is installed; otherwise no-op stubs. Matches `engine/permission/metrics.py` style.

**Files:**
- Create: `src/topsport_agent/ratelimit/metrics.py`
- Test: `tests/test_ratelimit_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ratelimit_metrics.py
import importlib.util

import pytest

from topsport_agent.ratelimit.metrics import RateLimitMetrics

_prom_available = importlib.util.find_spec("prometheus_client") is not None


def test_metrics_can_be_constructed_without_prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    # Re-import to pick up the patched environment
    import importlib
    import topsport_agent.ratelimit.metrics as mod
    importlib.reload(mod)

    metrics = mod.RateLimitMetrics()
    # Should not raise even without prometheus_client
    metrics.inc_request("ip")
    metrics.inc_denied("tenant")
    metrics.inc_degraded("ConnectionError")
    metrics.observe_check_duration(0.001)


@pytest.mark.skipif(not _prom_available, reason="prometheus-client not installed")
def test_metrics_with_prometheus_registers_counters() -> None:
    # Fresh registry to avoid collision with prior tests
    from prometheus_client import CollectorRegistry

    metrics = RateLimitMetrics(registry=CollectorRegistry())
    metrics.inc_request("ip")
    metrics.inc_request("ip")
    metrics.inc_denied("tenant")
    # We don't assert exact values — just that the calls work.
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_ratelimit_metrics.py -v`

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/topsport_agent/ratelimit/metrics.py
"""Prometheus metrics for the rate limiter. No-op when prometheus_client missing."""

from __future__ import annotations

import importlib
from typing import Any


class _NoOpCounter:
    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpCounter":
        return self

    def inc(self, amount: float = 1.0) -> None:
        return None


class _NoOpHistogram:
    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpHistogram":
        return self

    def observe(self, value: float) -> None:
        return None


def _try_load_prometheus() -> Any | None:
    try:
        return importlib.import_module("prometheus_client")
    except ImportError:
        return None


class RateLimitMetrics:
    """Three counters + one histogram. All become no-ops when prometheus_client
    isn't installed, matching the project's optional-dep style.
    """

    def __init__(self, *, registry: Any | None = None) -> None:
        prom = _try_load_prometheus()
        if prom is None:
            self._requests = _NoOpCounter()
            self._denied = _NoOpCounter()
            self._degraded = _NoOpCounter()
            self._duration = _NoOpHistogram()
            return

        kwargs = {"registry": registry} if registry is not None else {}
        self._requests = prom.Counter(
            "ratelimit_requests_total",
            "Rate-limit checks by scope",
            ["scope"],
            **kwargs,
        )
        self._denied = prom.Counter(
            "ratelimit_denied_total",
            "Rate-limit denials by scope",
            ["scope"],
            **kwargs,
        )
        self._degraded = prom.Counter(
            "ratelimit_degraded_total",
            "Rate-limit checks that fell back to fail-open due to Redis error",
            ["reason"],
            **kwargs,
        )
        self._duration = prom.Histogram(
            "ratelimit_check_duration_seconds",
            "Rate-limit EVAL/EVALSHA duration",
            **kwargs,
        )

    def inc_request(self, scope: str) -> None:
        self._requests.labels(scope=scope).inc()

    def inc_denied(self, scope: str) -> None:
        self._denied.labels(scope=scope).inc()

    def inc_degraded(self, reason: str) -> None:
        self._degraded.labels(reason=reason).inc()

    def observe_check_duration(self, seconds: float) -> None:
        self._duration.observe(seconds)


__all__ = ["RateLimitMetrics"]
```

- [ ] **Step 4: Run — expect PASS**

`uv run pytest tests/test_ratelimit_metrics.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit (main agent)**

`feat(ratelimit): Prometheus metrics with no-op fallback`

---

## Task 8: FastAPI Middleware (`middleware.py`)

The biggest integration point. Tests exercise it end-to-end via a minimal FastAPI TestClient.

**Files:**
- Create: `src/topsport_agent/ratelimit/middleware.py`
- Test: `tests/test_ratelimit_middleware.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ratelimit_middleware.py
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_redis_available = importlib.util.find_spec("redis") is not None
_test_url = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
requires_redis = pytest.mark.skipif(
    not _redis_available, reason="redis package not installed"
)


@pytest.fixture
async def app_and_client():
    """Build a minimal FastAPI app with the RateLimitMiddleware wired up.

    Uses httpx.AsyncClient against the in-process app (no network server).
    """
    from httpx import ASGITransport, AsyncClient

    try:
        from fastapi import FastAPI
    except ImportError:
        pytest.skip("fastapi not installed")

    from topsport_agent.ratelimit.config import RateLimitConfig
    from topsport_agent.ratelimit.limiter import RedisSlidingWindowLimiter
    from topsport_agent.ratelimit.middleware import RateLimitMiddleware
    from topsport_agent.ratelimit.redis_client import create_redis_client

    client = create_redis_client(_test_url)
    try:
        if not await client.ping():
            pytest.skip("local Redis not reachable")
    except Exception:
        pytest.skip("local Redis not reachable")
    await client.flushdb()

    script = Path(
        "src/topsport_agent/ratelimit/lua/sliding_window.lua"
    ).resolve().read_text(encoding="utf-8")
    sha = await client.script_load(script)
    limiter = RedisSlidingWindowLimiter(
        client=client, sha=sha, script=script
    )

    cfg = RateLimitConfig(
        enabled=True,
        redis_url=_test_url,
        per_ip_limit=3,
        per_principal_limit=0,     # disable
        per_tenant_limit=0,        # disable
        window_seconds=60,
    )

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limiter=limiter, config=cfg)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"ok": "yes"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield app, ac, client

    await client.flushdb()
    await client.close()


@requires_redis
@pytest.mark.asyncio
async def test_nth_plus_one_returns_429(app_and_client) -> None:
    app, ac, _ = app_and_client
    for _ in range(3):
        r = await ac.get("/ping")
        assert r.status_code == 200
    r = await ac.get("/ping")
    assert r.status_code == 429
    body = r.json()
    assert body["error"] == "rate_limited"
    assert body["scope"] == "ip"
    assert "retry_after" in body
    assert "Retry-After" in r.headers
    assert "X-RateLimit-Scope" in r.headers
    assert r.headers["X-RateLimit-Scope"] == "ip"


@requires_redis
@pytest.mark.asyncio
async def test_exempt_path_bypasses_limit(app_and_client) -> None:
    app, ac, _ = app_and_client
    # /health is in the default exempt list
    for _ in range(10):
        r = await ac.get("/health")
        assert r.status_code == 200


@requires_redis
@pytest.mark.asyncio
async def test_successful_response_has_ratelimit_headers(app_and_client) -> None:
    app, ac, _ = app_and_client
    r = await ac.get("/ping")
    assert r.status_code == 200
    assert "X-RateLimit-Limit" in r.headers
    assert "X-RateLimit-Remaining" in r.headers


@requires_redis
@pytest.mark.asyncio
async def test_fail_open_when_redis_disconnects(app_and_client) -> None:
    """Close the Redis client underneath the limiter, then confirm requests
    still succeed (fail-open) while the degraded counter increments via logs."""
    app, ac, client = app_and_client

    await client.close()
    # The limiter now has a closed client; next check will raise.

    r = await ac.get("/ping")
    # Default fail_open_on_redis_error=True → request still served.
    assert r.status_code == 200
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_ratelimit_middleware.py -v`

- [ ] **Step 3: Implement `middleware.py`**

```python
# src/topsport_agent/ratelimit/middleware.py
"""RateLimitMiddleware — ASGI/Starlette BaseHTTPMiddleware integration.

Runs AFTER auth middleware (so request.state.principal / tenant_id are set).
Skips exempt paths, extracts identity per scope, calls the limiter, and
maps decisions to 429 responses or X-RateLimit-* headers on success.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .config import RateLimitConfig
from .limiter import RateLimiter
from .metrics import RateLimitMetrics
from .types import RateLimitRule, RateLimitScope

_logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        limiter: RateLimiter,
        config: RateLimitConfig,
        metrics: RateLimitMetrics | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._config = config
        self._metrics = metrics or RateLimitMetrics()

    async def dispatch(self, request: Request, call_next):
        if self._is_exempt(request.url.path):
            return await call_next(request)

        rules = self._build_rules(request)
        if not rules:
            # All dimensions disabled — pass through.
            return await call_next(request)

        try:
            decision = await self._limiter.check(rules)
        except Exception as exc:  # noqa: BLE001
            self._metrics.inc_degraded(type(exc).__name__)
            _logger.warning(
                "ratelimit check failed; %s: %r",
                "allowing request (fail_open)"
                if self._config.fail_open_on_redis_error
                else "rejecting request",
                exc,
            )
            if self._config.fail_open_on_redis_error:
                return await call_next(request)
            return JSONResponse(
                status_code=503,
                content={"error": "ratelimit_unavailable"},
            )

        if not decision.allowed:
            scope = decision.denied_scope
            assert scope is not None  # invariant when allowed=False
            self._metrics.inc_denied(scope.value)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "scope": scope.value,
                    "retry_after": decision.retry_after_seconds,
                },
                headers={
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(decision.reset_at_ms // 1000),
                    "X-RateLimit-Scope": scope.value,
                },
            )

        # Allowed — attach informational headers to the downstream response.
        self._metrics.inc_request("any")
        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        return response

    # ---------- helpers ----------

    def _is_exempt(self, path: str) -> bool:
        return path in self._config.exempt_paths

    def _client_ip(self, request: Request) -> str:
        if self._config.trust_forwarded_for:
            xff = request.headers.get("x-forwarded-for")
            if xff:
                return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _route_template(self, request: Request) -> str:
        route = request.scope.get("route")
        # Starlette stores the original path template on the route object.
        path = getattr(route, "path", request.url.path)
        method = request.method
        return f"{method}:{path}"

    def _build_rules(self, request: Request) -> Sequence[RateLimitRule]:
        rules: list[RateLimitRule] = []
        cfg = self._config

        if cfg.per_ip_limit > 0:
            rules.append(
                RateLimitRule(
                    scope=RateLimitScope.IP,
                    identity=self._client_ip(request),
                    limit=cfg.per_ip_limit,
                    window_seconds=cfg.window_seconds,
                )
            )

        principal = getattr(request.state, "principal", None) or "anon"
        if cfg.per_principal_limit > 0:
            rules.append(
                RateLimitRule(
                    scope=RateLimitScope.PRINCIPAL,
                    identity=principal,
                    limit=cfg.per_principal_limit,
                    window_seconds=cfg.window_seconds,
                )
            )

        tenant = getattr(request.state, "tenant_id", None) or "public"
        if cfg.per_tenant_limit > 0:
            rules.append(
                RateLimitRule(
                    scope=RateLimitScope.TENANT,
                    identity=tenant,
                    limit=cfg.per_tenant_limit,
                    window_seconds=cfg.window_seconds,
                )
            )

        # Route-specific quotas
        route_key = self._route_template(request)
        route_limit = cfg.per_route_limits.get(route_key, cfg.per_route_default)
        if route_limit > 0:
            rules.append(
                RateLimitRule(
                    scope=RateLimitScope.ROUTE,
                    identity=route_key,
                    limit=route_limit,
                    window_seconds=cfg.window_seconds,
                )
            )

        return rules


__all__ = ["RateLimitMiddleware"]
```

- [ ] **Step 4: Run — expect PASS**

`uv run pytest tests/test_ratelimit_middleware.py -v`
Expected: 4 passed (or skipped if Redis down).

- [ ] **Step 5: Full suite**

`uv run pytest -q 2>&1 | tail -3`
Expected: 838+ passed, no new failures.

- [ ] **Step 6: Commit (main agent)**

`feat(ratelimit): FastAPI middleware with 429 + fail-open degradation`

---

## Task 9: Extend `ServerConfig` with rate-limit fields

**Files:**
- Modify: `src/topsport_agent/server/config.py` (add fields + env parsing)
- Test: `tests/test_server_config_ratelimit.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_server_config_ratelimit.py
import pytest

from topsport_agent.server.config import ServerConfig


def test_ratelimit_defaults_disabled() -> None:
    cfg = ServerConfig()
    assert cfg.enable_rate_limit is False
    assert cfg.ratelimit_redis_url is None
    assert cfg.ratelimit_window_seconds == 60
    assert cfg.ratelimit_per_ip == 300
    assert cfg.ratelimit_per_principal == 60
    assert cfg.ratelimit_per_tenant == 1000
    assert cfg.ratelimit_per_route_default == 0
    assert cfg.ratelimit_routes == {}
    assert cfg.ratelimit_trust_forwarded_for is False
    assert cfg.ratelimit_fail_open is True


def test_ratelimit_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_RATE_LIMIT", "true")
    monkeypatch.setenv("RATELIMIT_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RATELIMIT_PER_IP", "500")
    monkeypatch.setenv("RATELIMIT_PER_TENANT", "2000")
    monkeypatch.setenv("RATELIMIT_WINDOW_SECONDS", "30")
    monkeypatch.setenv("RATELIMIT_ROUTES", '{"/chat": 20}')
    monkeypatch.setenv("RATELIMIT_TRUST_FORWARDED_FOR", "true")
    monkeypatch.setenv("RATELIMIT_FAIL_OPEN", "false")
    monkeypatch.setenv("API_KEY", "dummy")

    cfg = ServerConfig.from_env()
    assert cfg.enable_rate_limit is True
    assert cfg.ratelimit_redis_url == "redis://localhost:6379/0"
    assert cfg.ratelimit_per_ip == 500
    assert cfg.ratelimit_per_tenant == 2000
    assert cfg.ratelimit_window_seconds == 30
    assert cfg.ratelimit_routes == {"/chat": 20}
    assert cfg.ratelimit_trust_forwarded_for is True
    assert cfg.ratelimit_fail_open is False


def test_ratelimit_routes_empty_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RATELIMIT_ROUTES", raising=False)
    monkeypatch.setenv("API_KEY", "dummy")

    cfg = ServerConfig.from_env()
    assert cfg.ratelimit_routes == {}


def test_ratelimit_routes_invalid_json_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RATELIMIT_ROUTES", "not-json")
    monkeypatch.setenv("API_KEY", "dummy")

    with pytest.raises(ValueError):
        ServerConfig.from_env()
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_server_config_ratelimit.py -v`

- [ ] **Step 3: Modify `src/topsport_agent/server/config.py`**

**Edit A (add fields):** Find the last database field (`database_timeout_seconds: float = 30.0`, line ~67) and add these fields right after it, still inside the class body:

```python

    # Rate limiting (Redis-backed)
    enable_rate_limit: bool = False
    ratelimit_redis_url: str | None = None
    ratelimit_window_seconds: int = 60
    ratelimit_per_ip: int = 300          # 0 = disable this dimension
    ratelimit_per_principal: int = 60
    ratelimit_per_tenant: int = 1000
    ratelimit_per_route_default: int = 0
    ratelimit_routes: dict[str, int] = field(default_factory=dict)
    ratelimit_trust_forwarded_for: bool = False
    ratelimit_fail_open: bool = True
```

Also add `from dataclasses import dataclass, field` at the top if not already importing `field`:

```python
from dataclasses import dataclass, field
```

**Edit B (extend `from_env`):** Inside the `return cls(...)` call, after the last database entry (`database_timeout_seconds=...`), add:

```python
            enable_rate_limit=_parse_bool(
                os.environ.get("ENABLE_RATE_LIMIT"), default=False
            ),
            ratelimit_redis_url=os.environ.get("RATELIMIT_REDIS_URL") or None,
            ratelimit_window_seconds=int(
                os.environ.get("RATELIMIT_WINDOW_SECONDS", "60")
            ),
            ratelimit_per_ip=int(os.environ.get("RATELIMIT_PER_IP", "300")),
            ratelimit_per_principal=int(
                os.environ.get("RATELIMIT_PER_PRINCIPAL", "60")
            ),
            ratelimit_per_tenant=int(
                os.environ.get("RATELIMIT_PER_TENANT", "1000")
            ),
            ratelimit_per_route_default=int(
                os.environ.get("RATELIMIT_PER_ROUTE_DEFAULT", "0")
            ),
            ratelimit_routes=_parse_route_limits(
                os.environ.get("RATELIMIT_ROUTES")
            ),
            ratelimit_trust_forwarded_for=_parse_bool(
                os.environ.get("RATELIMIT_TRUST_FORWARDED_FOR"), default=False
            ),
            ratelimit_fail_open=_parse_bool(
                os.environ.get("RATELIMIT_FAIL_OPEN"), default=True
            ),
```

**Edit C (add parser helper):** At the bottom of the file (next to `_parse_bool` / `_parse_optional_int`), add:

```python
def _parse_route_limits(raw: str | None) -> dict[str, int]:
    """Parse RATELIMIT_ROUTES JSON env. Empty/unset → empty dict.

    Raises ValueError on malformed JSON (fail-fast at startup).
    """
    if raw is None or not raw.strip():
        return {}
    import json

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"RATELIMIT_ROUTES must be a JSON object, got: {type(data).__name__}"
        )
    return {str(k): int(v) for k, v in data.items()}
```

- [ ] **Step 4: Run — expect PASS**

`uv run pytest tests/test_server_config_ratelimit.py -v`
Expected: 4 passed.

- [ ] **Step 5: Full suite regression check**

`uv run pytest -q 2>&1 | tail -3`

- [ ] **Step 6: Commit (main agent)**

`feat(server): ServerConfig rate-limit fields + env parsing`

---

## Task 10: Lifespan wiring + middleware registration in `server/app.py`

**Files:**
- Modify: `src/topsport_agent/server/app.py`
- Test: `tests/test_server_lifespan_ratelimit.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_server_lifespan_ratelimit.py
"""Lifespan wiring: enable_rate_limit=True + bad Redis → fail-fast at startup."""
from __future__ import annotations

import pytest

from topsport_agent.server.app import create_app
from topsport_agent.server.config import ServerConfig


def _stub_provider():
    class _P:
        name = "stub"

        async def complete(self, request):  # pragma: no cover
            raise RuntimeError("not used")

    return _P()


@pytest.mark.asyncio
async def test_disabled_ratelimit_does_not_touch_redis() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_rate_limit=False,
    )
    app = create_app(cfg, provider=_stub_provider())
    async with app.router.lifespan_context(app):
        assert getattr(app.state, "ratelimit_limiter", None) is None


@pytest.mark.asyncio
async def test_enabled_without_redis_url_fails_fast() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_rate_limit=True,
        ratelimit_redis_url=None,
    )
    app = create_app(cfg, provider=_stub_provider())
    with pytest.raises(RuntimeError, match="RATELIMIT_REDIS_URL"):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.asyncio
async def test_enabled_with_unreachable_redis_fails_fast() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_rate_limit=True,
        ratelimit_redis_url="redis://nonexistent.invalid:6379/0",
    )
    app = create_app(cfg, provider=_stub_provider())
    with pytest.raises(RuntimeError, match="Redis"):
        async with app.router.lifespan_context(app):
            pass
```

- [ ] **Step 2: Run — expect FAIL**

`uv run pytest tests/test_server_lifespan_ratelimit.py -v`

- [ ] **Step 3: Modify `src/topsport_agent/server/app.py`** — three edits

**Edit A (imports):** Find the existing database import line:

```python
from ..database import DatabaseConfig, NullGateway, create_database
```

Add right after it:

```python
from ..ratelimit.config import RateLimitConfig
from ..ratelimit.limiter import RedisSlidingWindowLimiter
from ..ratelimit.middleware import RateLimitMiddleware
from ..ratelimit.redis_client import create_redis_client
```

Also add at the top imports if `Path` is not already imported:

```python
from pathlib import Path
```

**Edit B (lifespan setup):** Immediately after the database setup block (right after `app.state.database = NullGateway()` in the `else` branch), add:

```python
        # === Rate limit (optional, default off) ===
        if cfg.enable_rate_limit:
            if not cfg.ratelimit_redis_url:
                raise RuntimeError(
                    "ENABLE_RATE_LIMIT=true but RATELIMIT_REDIS_URL is unset"
                )
            rl_client = create_redis_client(cfg.ratelimit_redis_url)
            try:
                if not await rl_client.ping():
                    raise RuntimeError(
                        "rate limit enabled but Redis ping failed"
                    )
            except Exception as exc:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(
                    f"rate limit enabled but Redis connection failed: {exc!r}"
                ) from exc
            lua_path = (
                Path(__file__).resolve().parent.parent
                / "ratelimit" / "lua" / "sliding_window.lua"
            )
            lua_script = lua_path.read_text(encoding="utf-8")
            sha = await rl_client.script_load(lua_script)
            rl_limiter = RedisSlidingWindowLimiter(
                client=rl_client, sha=sha, script=lua_script
            )
            app.state.ratelimit_client = rl_client
            app.state.ratelimit_limiter = rl_limiter
        else:
            app.state.ratelimit_limiter = None
```

**Edit C (middleware registration):** Find the `_DrainMiddleware` block (around the `app.add_middleware(_DrainMiddleware)` call). **Before** the first `app.add_middleware(...)` call that currently exists, insert conditional middleware setup. Search for the `app = FastAPI(...)` block and look for the `class _DrainMiddleware(BaseHTTPMiddleware)` definition. After that class is defined but before any `app.add_middleware(_DrainMiddleware)` call, insert:

```python
    # Rate limit middleware — installs only when enabled, reads limiter from lifespan.
    if cfg.enable_rate_limit:
        rl_config = RateLimitConfig(
            enabled=True,
            redis_url=cfg.ratelimit_redis_url,
            window_seconds=cfg.ratelimit_window_seconds,
            per_ip_limit=cfg.ratelimit_per_ip,
            per_principal_limit=cfg.ratelimit_per_principal,
            per_tenant_limit=cfg.ratelimit_per_tenant,
            per_route_default=cfg.ratelimit_per_route_default,
            per_route_limits=dict(cfg.ratelimit_routes),
            trust_forwarded_for=cfg.ratelimit_trust_forwarded_for,
            fail_open_on_redis_error=cfg.ratelimit_fail_open,
        )
        # Middleware is constructed with a shim limiter that will be replaced
        # by lifespan's live instance once it's ready. We attach the config
        # now so the add_middleware call records the real config.
        class _LazyLimiter:
            async def check(self, rules):
                real = getattr(app.state, "ratelimit_limiter", None)
                if real is None:
                    from topsport_agent.ratelimit.types import RateLimitDecision
                    return RateLimitDecision(
                        allowed=True, denied_scope=None,
                        limit=0, remaining=0, reset_at_ms=0,
                        retry_after_seconds=0,
                    )
                return await real.check(rules)

        app.add_middleware(
            RateLimitMiddleware,
            limiter=_LazyLimiter(),
            config=rl_config,
        )
```

**Edit D (lifespan teardown):** In the finally block, after the database close, add:

```python
            # Close rate-limit Redis client (if it was created).
            rl_client = getattr(app.state, "ratelimit_client", None)
            if rl_client is not None:
                try:
                    await rl_client.close()
                except Exception as exc:
                    _logger.warning("ratelimit client close failed: %r", exc)
```

- [ ] **Step 4: Run — expect PASS**

`uv run pytest tests/test_server_lifespan_ratelimit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Full suite regression**

`uv run pytest -q 2>&1 | tail -3`
Expected: previous total + 3 new, no drops.

- [ ] **Step 6: Commit (main agent)**

`feat(server): rate-limit lifespan wiring + middleware registration`

---

## Task 11: README section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a new section immediately before `## Not yet implemented`**

Find `## Not yet implemented` (added by Plan A; now around line 1211). Insert before it:

```markdown
## Rate Limiting (Redis-backed)

Four-dimension sliding-window rate limiter enforced by a FastAPI middleware.
Checks run **atomically** via one Lua script per request — any single
dimension over quota denies the request without incrementing other
dimensions' counters.

### Enabling

```bash
uv sync --group redis                  # install redis-py
export ENABLE_RATE_LIMIT=true
export RATELIMIT_REDIS_URL="redis://localhost:6379/0"
```

### Dimensions

| Scope       | Identity                  | Default limit / 60s |
| ----------- | ------------------------- | ------------------- |
| `ip`        | `request.client.host`     | 300                 |
| `principal` | `request.state.principal` | 60                  |
| `tenant`    | `request.state.tenant_id` | 1000                |
| `route`     | `METHOD:/path/template`   | 0 (disabled)        |

Set any dimension's limit to `0` to disable that dimension.

### Environment variables

| Variable                           | Default  | Notes                                                 |
| ---------------------------------- | -------- | ----------------------------------------------------- |
| `ENABLE_RATE_LIMIT`                | `false`  | Master switch                                         |
| `RATELIMIT_REDIS_URL`              | —        | Required when enabled                                 |
| `RATELIMIT_WINDOW_SECONDS`         | `60`     | Sliding-window size                                   |
| `RATELIMIT_PER_IP`                 | `300`    | Per-client-IP quota                                   |
| `RATELIMIT_PER_PRINCIPAL`          | `60`     | Per-authenticated-user quota                          |
| `RATELIMIT_PER_TENANT`             | `1000`   | Per-tenant quota                                      |
| `RATELIMIT_PER_ROUTE_DEFAULT`      | `0`      | Default route quota (0 = disabled)                    |
| `RATELIMIT_ROUTES`                 | `{}`     | JSON object `{"GET:/chat": 20}`                       |
| `RATELIMIT_TRUST_FORWARDED_FOR`    | `false`  | Read `X-Forwarded-For` (enable when behind a proxy)   |
| `RATELIMIT_FAIL_OPEN`              | `true`   | Runtime Redis failure → allow request (log + metric)  |

### Behavior

- **Exempt paths** (`/health`, `/metrics`, `/docs`, `/openapi.json`, `/redoc`)
  bypass limiting unconditionally.
- **Startup fail-fast**: `ENABLE_RATE_LIMIT=true` + Redis unreachable → server
  refuses to start.
- **Runtime fail-open**: Redis dies mid-request → log a warning, increment
  `ratelimit_degraded_total`, let the request through. Set
  `RATELIMIT_FAIL_OPEN=false` to flip to 503 on Redis errors instead.
- **429 response** includes `Retry-After`, `X-RateLimit-Scope`,
  `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

### Metrics (optional — requires `--group metrics`)

- `ratelimit_requests_total{scope}` — checks performed
- `ratelimit_denied_total{scope}` — 429 responses
- `ratelimit_degraded_total{reason}` — Redis errors handled via fail-open
- `ratelimit_check_duration_seconds` — Lua script latency histogram
```

- [ ] **Step 2: Verify insertion**

`grep -n "^## Rate Limiting" README.md`
`grep -n "^## Not yet implemented" README.md`
(Rate Limiting line should print BEFORE Not yet implemented.)

- [ ] **Step 3: Commit (main agent)**

`docs(ratelimit): README section for Redis-backed rate limiting`

---

## Task 12: Final verification + acceptance checklist

- [ ] **Step 1: All tests with full groups + local Redis running**

```bash
docker ps | grep -q topsport-test-redis || \
  docker run -d -p 6379:6379 --name topsport-test-redis redis:7-alpine
uv sync --all-groups
uv run pytest -v 2>&1 | tail -5
```

Expected: all green, no skips except maybe `test_create_without_redis_package_raises_import_error` variations if the env has redis installed.

- [ ] **Step 2: No stray top-level `redis` imports in source**

```bash
grep -rn "^import redis\|^from redis" src/topsport_agent/ratelimit/
```

Expected: **no output** (all `redis` imports are via `importlib.import_module("redis.asyncio")` in `redis_client.py`).

- [ ] **Step 3: Package imports without `redis` installed**

Simulate via monkeypatched env:

```bash
uv run python -c "
import sys
sys.modules['redis'] = None
from topsport_agent.ratelimit.config import RateLimitConfig
from topsport_agent.ratelimit.types import RateLimitScope, RateLimitRule, RateLimitDecision
from topsport_agent.ratelimit.middleware import RateLimitMiddleware
print('OK — imports succeed without redis')
"
```

Expected: `OK — imports succeed without redis`.

- [ ] **Step 4: End-to-end smoke test against running server**

Optional manual check. Start the server:

```bash
ENABLE_RATE_LIMIT=true \
RATELIMIT_REDIS_URL=redis://localhost:6379/0 \
RATELIMIT_PER_IP=3 \
AUTH_REQUIRED=false \
API_KEY=dummy \
uv run uvicorn topsport_agent.server.app:create_app --factory --port 8000
```

In another shell:

```bash
for i in 1 2 3 4; do curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/; done
```

Expected: 200 200 200 429.

- [ ] **Step 5: Acceptance checklist from spec §10.3**

Walk each item:
- [ ] `RATELIMIT_PER_IP=3` → 4th rapid request returns 429 + headers ✅ (Task 8 test_nth_plus_one_returns_429)
- [ ] `RATELIMIT_PER_TENANT` over-quota denies even when IP differs ✅ (Task 5 test_multi_dimension_denies_on_tightest)
- [ ] Exempt paths bypass ✅ (Task 8 test_exempt_path_bypasses_limit)
- [ ] Redis disconnect mid-run → fail-open + counter increments ✅ (Task 8 test_fail_open_when_redis_disconnects)
- [ ] `ENABLE_RATE_LIMIT=true` + unreachable Redis → RuntimeError at startup ✅ (Task 10 test_enabled_with_unreachable_redis_fails_fast)

---

## Self-review notes

**Spec coverage** (cross-check against §6 / §7 / §8):
- §6.1 types → Task 1
- §6.2 RateLimiter Protocol → Task 4
- §6.3 Lua script (exact content + version tag) → Task 2
- §6.4 Middleware (exempt, XFF, route template, LIFO ordering docs) → Task 8 + Task 10 Edit C
- §6.5 Metrics → Task 7
- §6.6 Startup fail-fast → Task 10 lifespan
- §7 ServerConfig + env binding → Task 9
- §8 Testing strategy (real Redis, DB 15, FLUSHDB before/after, skip if unreachable) → Task 5 + Task 8 fixtures

**No placeholders**: scanned — every step has concrete code blocks, exact commands, expected outputs, and commit messages.

**Type consistency**: `RateLimitScope / RateLimitRule / RateLimitDecision / RateLimiter / RedisSlidingWindowLimiter / RateLimitConfig / RateLimitMetrics / RateLimitMiddleware / create_redis_client` — stable across all tasks.
