"""Session <-> OpenSandbox 沙箱实例的绑定池。

行为约束：
- 同一 session_id 在生命周期内只对应一个 sandbox（lazy 首创 + 复用）
- 并发 acquire 同一 session 只触发一次 factory（per-session asyncio.Lock）
- factory 失败不缓存，下次重试
- release 幂等，kill 异常被吞并记 warning，不影响 pool 一致性
- close_all 兜底清理全部 session

可选能力（按需开启）：
- idle_pause_seconds：空闲超过阈值自动 pause，acquire 时自动 resume，
  节省内存而不丢 session 状态（基于 SDK 的 pause/resume）
- http_client_factory：共享 httpx.AsyncClient，供 fast_exec 复用连接池
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

_logger = logging.getLogger(__name__)

SandboxFactory = Callable[[str], Awaitable[Any]]
SandboxResumeFactory = Callable[[Any], Awaitable[Any]]


class TenantQuotaExceeded(RuntimeError):
    """租户并发沙箱数超 per_tenant_max_sandboxes 且在超时内未释放出空位。"""

    def __init__(self, tenant_id: str, limit: int) -> None:
        super().__init__(
            f"tenant {tenant_id!r} reached concurrent sandbox limit={limit}"
        )
        self.tenant_id = tenant_id
        self.limit = limit


class OpenSandboxPool:
    def __init__(
        self,
        *,
        sandbox_factory: SandboxFactory,
        http_client_factory: Callable[[], Any] | None = None,
        sandbox_resume_factory: SandboxResumeFactory | None = None,
        idle_pause_seconds: float | None = None,
        reaper_interval_seconds: float = 30.0,
        per_tenant_max_sandboxes: int | None = None,
        per_tenant_acquire_timeout: float | None = None,
    ) -> None:
        """
        sandbox_factory: 按 session_id 异步构造一个 Sandbox。
        http_client_factory: 按需创建 httpx.AsyncClient；供 fast_exec 用。
        sandbox_resume_factory: 从已 pause 的 sandbox 恢复；None 时走默认
          opensandbox.Sandbox.resume 路径。
        idle_pause_seconds: 空闲超阈值 reaper 调 sandbox.pause()；None = 禁用。
        reaper_interval_seconds: reaper 扫描间隔。只在 idle_pause_seconds 启用时生效。
        per_tenant_max_sandboxes: 单租户并发沙箱上限。None = 不限。
        per_tenant_acquire_timeout: 租户 quota 满时等待释放的秒数。
          None = 阻塞到有空位；>0 超时抛 TenantQuotaExceeded。
        """
        self._factory = sandbox_factory
        self._http_client_factory = http_client_factory
        self._resume_factory = sandbox_resume_factory
        self._handles: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_guard = asyncio.Lock()
        self._http_client: Any | None = None
        self._http_client_lock = asyncio.Lock()
        # idle pause 相关状态
        self._idle_pause_seconds = idle_pause_seconds
        self._reaper_interval_seconds = reaper_interval_seconds
        self._last_used_at: dict[str, float] = {}
        self._paused: dict[str, bool] = {}
        self._reaper_task: asyncio.Task[None] | None = None
        # per-tenant 配额
        self._per_tenant_max = per_tenant_max_sandboxes
        self._per_tenant_timeout = per_tenant_acquire_timeout
        self._tenant_of: dict[str, str | None] = {}
        self._tenant_sems: dict[str, asyncio.Semaphore] = {}
        self._tenant_lock = asyncio.Lock()
        # 该 session 是否已消耗了 tenant 信号量（防 release 重复放）
        self._sem_held: dict[str, bool] = {}
        # prefix → tenant 映射：用于 plan 路径等 "session_id 由下游动态拼接" 的场景。
        # acquire 时若 session_id 无显式 binding，按最长 prefix 匹配推断 tenant。
        self._tenant_prefixes: dict[str, str | None] = {}

    def bind_tenant(self, session_id: str, tenant_id: str | None) -> None:
        """绑定 session 到 tenant。同步；在任何 acquire 之前调用。

        后续 acquire 按 tenant_id 取信号量；release 时按同 tenant 释放。
        重复绑定到相同 tenant 是 no-op；重绑到不同 tenant 会抛（避免配额核对串位）。
        """
        prev = self._tenant_of.get(session_id)
        if session_id in self._tenant_of and prev != tenant_id:
            raise ValueError(
                f"session {session_id!r} already bound to tenant {prev!r}, "
                f"cannot rebind to {tenant_id!r}"
            )
        self._tenant_of[session_id] = tenant_id

    def bind_tenant_prefix(self, prefix: str, tenant_id: str | None) -> None:
        """按前缀绑定 tenant：session_id 以 prefix 开头且未显式 bind 时走该 tenant。

        用于 plan 路径：plan_execute 预先 bind_tenant_prefix(f"{plan_id}:", principal)，
        之后 orchestrator.spawn_child 拼出的 session_id（形如 {plan_id}:{step_id}:{uuid}）
        在首次 acquire 时被自动归到该 tenant，受 per_tenant_max_sandboxes 配额约束。

        `release_by_prefix(prefix)` 会顺带清除该 prefix 绑定。
        """
        if not prefix:
            raise ValueError("prefix must be non-empty")
        self._tenant_prefixes[prefix] = tenant_id

    def _resolve_tenant(self, session_id: str) -> str | None:
        """返回 session_id 归属的 tenant_id，按 explicit > 最长 prefix 顺序。"""
        if session_id in self._tenant_of:
            return self._tenant_of[session_id]
        # 按最长 prefix 优先，避免 "plan-1" / "plan-1:" 同时存在时错配
        best_prefix: str | None = None
        for prefix in self._tenant_prefixes:
            if session_id.startswith(prefix):
                if best_prefix is None or len(prefix) > len(best_prefix):
                    best_prefix = prefix
        return self._tenant_prefixes[best_prefix] if best_prefix is not None else None

    async def acquire(
        self,
        session_id: str,
        *,
        tenant_id: str | None = None,
    ) -> Any:
        """返回 session 绑定的 sandbox；首次调用同步创建，后续复用。

        tenant_id：如传入且之前未绑定，等同于 bind_tenant；若已绑定必须一致。
        行为：
        - 首次 acquire：按 tenant 的 semaphore 等位，然后 factory。
        - 若 handle 存在但被 reaper pause：resume 替换 handle。
        - 每次 acquire 更新 last_used_at（供 reaper 判空闲）。
        """
        if tenant_id is not None:
            self.bind_tenant(session_id, tenant_id)
        # explicit bind > prefix 推断；首次 acquire 时把 prefix 推断的结果
        # 固化到 _tenant_of，避免同一 session 后续多次命中不同 prefix。
        effective_tenant = self._resolve_tenant(session_id)
        if session_id not in self._tenant_of and effective_tenant is not None:
            self._tenant_of[session_id] = effective_tenant

        lock = await self._get_lock(session_id)
        async with lock:
            existing = self._handles.get(session_id)
            if existing is not None:
                if self._paused.get(session_id):
                    resumed = await self._resume(existing)
                    self._handles[session_id] = resumed
                    self._paused[session_id] = False
                    self._last_used_at[session_id] = time.monotonic()
                    self._ensure_reaper_started()
                    return resumed
                self._last_used_at[session_id] = time.monotonic()
                self._ensure_reaper_started()
                return existing

            # 首次创建：先过 tenant quota（在 per-session lock 内做等待也可，
            # 因为 per-session lock 不会跨 session 阻塞）
            acquired_sem = False
            if effective_tenant is not None and self._per_tenant_max is not None:
                sem = await self._get_tenant_sem(effective_tenant)
                try:
                    if self._per_tenant_timeout is None:
                        await sem.acquire()
                    else:
                        await asyncio.wait_for(
                            sem.acquire(), timeout=self._per_tenant_timeout
                        )
                except TimeoutError as exc:
                    raise TenantQuotaExceeded(
                        effective_tenant, self._per_tenant_max
                    ) from exc
                acquired_sem = True

            try:
                sandbox = await self._factory(session_id)
            except Exception:
                if acquired_sem and effective_tenant is not None:
                    # 创建失败不占配额
                    self._tenant_sems[effective_tenant].release()
                raise

            self._handles[session_id] = sandbox
            self._paused[session_id] = False
            self._last_used_at[session_id] = time.monotonic()
            self._sem_held[session_id] = acquired_sem
            self._ensure_reaper_started()
            return sandbox

    async def release(self, session_id: str) -> None:
        """关闭并移除指定 session 的 sandbox。幂等；kill 异常被吞。

        若 sandbox 处于 paused 状态也照样 kill（SDK 允许对 paused sandbox kill）。
        释放 tenant 信号量（只对消耗过 semaphore 的 session）。
        """
        sandbox = self._handles.pop(session_id, None)
        self._last_used_at.pop(session_id, None)
        self._paused.pop(session_id, None)
        sem_held = self._sem_held.pop(session_id, False)
        tenant = self._tenant_of.pop(session_id, None)
        if sem_held and tenant is not None:
            sem = self._tenant_sems.get(tenant)
            if sem is not None:
                sem.release()
        if sandbox is None:
            return
        try:
            await sandbox.kill()
        except Exception as exc:
            _logger.warning(
                "sandbox kill failed for session=%s: %r", session_id, exc
            )

    async def release_by_prefix(self, prefix: str) -> int:
        """批量释放所有 session_id 以 prefix 开头的沙箱，并清理 prefix 绑定。

        用于 plan 路径：orchestrator 为每个 step 创建 `f"{plan_id}:{step_id}"` 形
        式的 session_id，plan 结束时按 `f"{plan_id}:"` 前缀统一清理；否则这些
        sandbox 要等 sandbox_timeout（默认 30min）才被 OpenSandbox 自动回收。

        返回实际被释放的 session 数量。幂等；对空集合返回 0。
        同步清掉 self._tenant_prefixes[prefix]（若存在），避免内存泄漏。
        """
        if not prefix:
            return 0
        matched = [sid for sid in self._handles if sid.startswith(prefix)]
        for sid in matched:
            await self.release(sid)
        # 清除 prefix → tenant 绑定（不再需要）
        self._tenant_prefixes.pop(prefix, None)
        return len(matched)

    async def close_all(self) -> None:
        """清理全部 sandbox 并关闭共享 http client + reaper。幂等。"""
        await self._stop_reaper()
        for sid in list(self._handles.keys()):
            await self.release(sid)
        client = self._http_client
        self._http_client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception as exc:
                _logger.warning("shared http client close failed: %r", exc)

    async def get_http_client(self) -> Any:
        """返回 pool 共享的 httpx.AsyncClient（lazy 创建）。"""
        if self._http_client is not None:
            return self._http_client
        async with self._http_client_lock:
            if self._http_client is not None:
                return self._http_client
            if self._http_client_factory is not None:
                self._http_client = self._http_client_factory()
            else:
                httpx_mod = importlib.import_module("httpx")
                self._http_client = httpx_mod.AsyncClient()
            return self._http_client

    def has(self, session_id: str) -> bool:
        return session_id in self._handles

    def is_paused(self, session_id: str) -> bool:
        """测试/观测用：sandbox 是否处于 reaper 置的 paused 状态。"""
        return bool(self._paused.get(session_id))

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        async with self._lock_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock

    async def _get_tenant_sem(self, tenant_id: str) -> asyncio.Semaphore:
        assert self._per_tenant_max is not None
        async with self._tenant_lock:
            sem = self._tenant_sems.get(tenant_id)
            if sem is None:
                sem = asyncio.Semaphore(self._per_tenant_max)
                self._tenant_sems[tenant_id] = sem
            return sem

    async def _resume(self, old_sandbox: Any) -> Any:
        """恢复一个 paused sandbox，返回新的 sandbox 句柄。"""
        if self._resume_factory is not None:
            return await self._resume_factory(old_sandbox)
        # 默认：走 opensandbox.Sandbox.resume
        opensandbox = importlib.import_module("opensandbox")
        Sandbox = opensandbox.Sandbox
        sid = getattr(old_sandbox, "id", None)
        conn_cfg = getattr(old_sandbox, "connection_config", None)
        if sid is None:
            raise RuntimeError("cannot resume sandbox: missing sandbox id")
        return await Sandbox.resume(sandbox_id=sid, connection_config=conn_cfg)

    def _ensure_reaper_started(self) -> None:
        """在第一次 acquire 后且启用 idle pause 时启动后台 reaper。

        用 create_task，需要当前在 event loop 里。构造函数不启动以支持
        pool 在非 async 上下文构造。
        """
        if self._idle_pause_seconds is None:
            return
        if self._reaper_task is not None and not self._reaper_task.done():
            return
        self._reaper_task = asyncio.create_task(self._reap_idle_loop())

    async def _stop_reaper(self) -> None:
        task = self._reaper_task
        self._reaper_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            # 取消或 reaper 自身异常都不抛给调用方
            pass

    async def _reap_idle_loop(self) -> None:
        """后台循环：周期性扫描，空闲 session 的 sandbox 调 pause()。"""
        assert self._idle_pause_seconds is not None
        while True:
            try:
                await asyncio.sleep(self._reaper_interval_seconds)
                await self._reap_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # reaper 不能因一次异常退出（会丢失全部 pool 的自动 pause）
                _logger.warning("reaper iteration failed: %r", exc)

    async def _reap_once(self) -> None:
        assert self._idle_pause_seconds is not None
        now = time.monotonic()
        threshold = self._idle_pause_seconds
        # 复制 key 列表避免遍历时并发修改
        for sid in list(self._handles.keys()):
            last = self._last_used_at.get(sid)
            if last is None:
                continue
            if now - last < threshold:
                continue
            if self._paused.get(sid):
                continue
            lock = await self._get_lock(sid)
            async with lock:
                # 锁内重查：可能在等锁期间刚刚被 acquire 更新或 release
                if sid not in self._handles:
                    continue
                if self._paused.get(sid):
                    continue
                last = self._last_used_at.get(sid, now)
                if now - last < threshold:
                    continue
                try:
                    await self._handles[sid].pause()
                    self._paused[sid] = True
                except Exception as exc:
                    _logger.warning(
                        "sandbox pause failed for session=%s: %r", sid, exc
                    )

    @classmethod
    def from_config(
        cls,
        *,
        domain: str,
        image: str = "ubuntu",
        api_key: str | None = None,
        sandbox_timeout_seconds: float = 1800.0,
        skip_health_check: bool = False,
        idle_pause_seconds: float | None = 300.0,
        reaper_interval_seconds: float = 30.0,
        per_tenant_max_sandboxes: int | None = None,
        per_tenant_acquire_timeout: float | None = None,
        **connection_kwargs: Any,
    ) -> "OpenSandboxPool":
        """生产路径：懒加载 opensandbox 构造真实 factory。

        默认开启 idle pause（300s）。设为 None 禁用。
        per_tenant_max_sandboxes：多租户场景建议显式设置（如 10），默认 None 不限。
        """
        from datetime import timedelta

        mod_name = "opensandbox"
        opensandbox = importlib.import_module(mod_name)
        config_mod = importlib.import_module("opensandbox.config")
        Sandbox = opensandbox.Sandbox
        ConnectionConfig = config_mod.ConnectionConfig
        conn_cfg = ConnectionConfig(
            api_key=api_key, domain=domain, **connection_kwargs
        )

        async def factory(session_id: str) -> Any:
            return await Sandbox.create(
                image,
                connection_config=conn_cfg,
                timeout=timedelta(seconds=sandbox_timeout_seconds),
                metadata={
                    "session_id": session_id,
                    "managed_by": "topsport-agent",
                },
                skip_health_check=skip_health_check,
            )

        return cls(
            sandbox_factory=factory,
            idle_pause_seconds=idle_pause_seconds,
            reaper_interval_seconds=reaper_interval_seconds,
            per_tenant_max_sandboxes=per_tenant_max_sandboxes,
            per_tenant_acquire_timeout=per_tenant_acquire_timeout,
        )
