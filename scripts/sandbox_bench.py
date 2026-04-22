# pyright: reportMissingImports=false
"""OpenSandbox 性能基线测试。

决策用途：
- 冷启动分布 → 判断是否需要 pool 预热 / session-scoped 复用
- Steady tool call 延迟 → 判断工具链路是否健康、出错时能否早期识别
- Pause/Resume 可行性 → 判断 session-scoped 策略的资源节约收益

输出 min / p50 / p95 / max（n=10 create+kill, n=20 steady shell）。
不并发，避免测量扰动。

用法：`uv run python scripts/sandbox_bench.py`
"""
from __future__ import annotations

import asyncio
import statistics
import time
from datetime import timedelta

import httpx
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig

from topsport_agent.sandbox.fast_exec import fast_run_command


def _summary(name: str, samples: list[float]) -> str:
    if not samples:
        return f"{name}: (no samples)"
    n = len(samples)
    mn = min(samples)
    mx = max(samples)
    p50 = statistics.median(samples)
    # p95 简单插值
    sorted_s = sorted(samples)
    idx = int(0.95 * (n - 1))
    p95 = sorted_s[idx]
    mean = statistics.mean(samples)
    return (
        f"{name:<24} n={n:<3} "
        f"min={mn*1000:>7.1f}ms  "
        f"p50={p50*1000:>7.1f}ms  "
        f"p95={p95*1000:>7.1f}ms  "
        f"max={mx*1000:>7.1f}ms  "
        f"mean={mean*1000:>7.1f}ms"
    )


async def bench_cold_start(cfg: ConnectionConfig, n: int) -> tuple[list[float], list[float]]:
    """10 次独立 create + kill，测冷启动与回收延迟。"""
    creates: list[float] = []
    kills: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        sb = await Sandbox.create("ubuntu", connection_config=cfg)
        creates.append(time.perf_counter() - t0)
        print(f"  [cold {i+1}/{n}] create={creates[-1]*1000:.0f}ms id={sb.id[:8]}")
        t1 = time.perf_counter()
        await sb.kill()
        kills.append(time.perf_counter() - t1)
    return creates, kills


async def bench_steady(cfg: ConnectionConfig, n: int) -> dict[str, list[float]]:
    """单沙箱跑 n 次工具调用，覆盖：SDK shell、fast_exec shell、files。"""
    sb = await Sandbox.create("ubuntu", connection_config=cfg)
    print(f"  [steady] sandbox created id={sb.id[:8]}")
    results: dict[str, list[float]] = {
        "shell_sdk": [], "shell_fast": [], "file_write": [], "file_read": []
    }
    http_client = httpx.AsyncClient()
    try:
        # warmup
        await sb.commands.run("echo warmup")
        await fast_run_command(sb, "echo warmup", httpx_client=http_client)

        for _ in range(n):
            t0 = time.perf_counter()
            await sb.commands.run("echo hi")
            results["shell_sdk"].append(time.perf_counter() - t0)

        for _ in range(n):
            t0 = time.perf_counter()
            await fast_run_command(sb, "echo hi", httpx_client=http_client)
            results["shell_fast"].append(time.perf_counter() - t0)

        from opensandbox.models.filesystem import WriteEntry
        for i in range(n):
            t0 = time.perf_counter()
            await sb.files.write_files([
                WriteEntry(path=f"/tmp/bench_{i}.txt", data=f"line-{i}")
            ])
            results["file_write"].append(time.perf_counter() - t0)

        for i in range(n):
            t0 = time.perf_counter()
            await sb.files.read_file(f"/tmp/bench_{i}.txt")
            results["file_read"].append(time.perf_counter() - t0)
    finally:
        await http_client.aclose()
        await sb.kill()
    return results


async def bench_pause_resume(cfg: ConnectionConfig) -> tuple[float, float, bool]:
    """测 pause + resume：对 session-scoped 策略意义重大。
    返回 (pause_sec, resume_sec, state_preserved)。
    """
    sb = await Sandbox.create("ubuntu", connection_config=cfg)
    sid = sb.id
    # 在沙箱里写一个标记文件
    from opensandbox.models.filesystem import WriteEntry
    await sb.files.write_files([WriteEntry(path="/tmp/mark.txt", data="pre-pause")])

    t0 = time.perf_counter()
    try:
        await sb.pause()
        pause_sec = time.perf_counter() - t0
    except Exception as exc:
        print(f"  [pause] NOT SUPPORTED: {type(exc).__name__}: {exc}")
        await sb.kill()
        return (-1.0, -1.0, False)

    t1 = time.perf_counter()
    try:
        sb2 = await Sandbox.resume(sandbox_id=sid, connection_config=cfg)
        resume_sec = time.perf_counter() - t1
    except Exception as exc:
        print(f"  [resume] FAILED: {type(exc).__name__}: {exc}")
        return (pause_sec, -1.0, False)

    # 验证状态保留
    try:
        content = await sb2.files.read_file("/tmp/mark.txt")
        preserved = (str(content).strip() == "pre-pause")
    except Exception as exc:
        print(f"  [resume] state read failed: {type(exc).__name__}: {exc}")
        preserved = False
    finally:
        await sb2.kill()

    return (pause_sec, resume_sec, preserved)


async def main() -> None:
    cfg = ConnectionConfig(
        domain="localhost:8090",
        protocol="http",
        use_server_proxy=True,
        request_timeout=timedelta(seconds=120),
    )
    print("=== cold start (create + kill) x 10 ===")
    creates, kills = await bench_cold_start(cfg, 10)

    print("\n=== steady tool calls (single sandbox) x 20 per type ===")
    steady = await bench_steady(cfg, 20)

    print("\n=== pause + resume (x 1) ===")
    pause_sec, resume_sec, preserved = await bench_pause_resume(cfg)

    print("\n=== SUMMARY ===")
    print(_summary("cold.create", creates))
    print(_summary("cold.kill", kills))
    print(_summary("steady.shell_sdk", steady["shell_sdk"]))
    print(_summary("steady.shell_fast", steady["shell_fast"]))
    print(_summary("steady.file_write", steady["file_write"]))
    print(_summary("steady.file_read", steady["file_read"]))
    if steady["shell_sdk"] and steady["shell_fast"]:
        p50_sdk = statistics.median(steady["shell_sdk"]) * 1000
        p50_fast = statistics.median(steady["shell_fast"]) * 1000
        print(f"\n  shell speedup (p50): {p50_sdk / p50_fast:.1f}x  ({p50_sdk:.0f}ms -> {p50_fast:.0f}ms)")
    print()
    if pause_sec < 0:
        print("pause/resume       NOT SUPPORTED (see above)")
    else:
        print(
            f"pause/resume       pause={pause_sec*1000:.0f}ms  "
            f"resume={resume_sec*1000:.0f}ms  state_preserved={preserved}"
        )


if __name__ == "__main__":
    asyncio.run(main())
