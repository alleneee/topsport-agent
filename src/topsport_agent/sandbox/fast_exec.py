"""绕过 opensandbox SDK 的 commands.run，早退 SSE 流避免 1s 阻塞。

根因（验证见 scripts/sandbox_bench.py 时序）：
  execd 的 /command SSE 端点在发出 `execution_complete` 事件后还会保持
  ~1 秒再关闭连接。SDK `_execute_streaming_request` 用
  `async for line in aiter_lines()` 阻塞到流关，导致即使 `echo hi` 也要 1 秒。

本模块直接调 execd 的 /command 端点，解析 SSE 事件行，
一旦看到 `execution_complete` 或 `error` 就 break，延迟从 ~1000ms 降到 ~15ms。

不替换 SDK 的其他能力（files / pause / resume / kill 仍走 SDK）。
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any

import httpx

# 与 SDK 内部常量对齐；这些是 openapi-python-client 生成后的稳定路径。
_RUN_COMMAND_PATH = "/command"


@dataclass(slots=True)
class FastExecResult:
    """fast_run_command 的返回值；字段和 SDK Execution 概念一致但扁平化。"""
    exit_code: int | None
    stdout: str
    stderr: str
    error: str | None


async def fast_run_command(
    sandbox: Any,
    command: str,
    *,
    httpx_client: httpx.AsyncClient,
    connect_timeout: float = 10.0,
) -> FastExecResult:
    """执行 shell 命令；收到 execution_complete/error 立即返回。

    sandbox: opensandbox.Sandbox 实例，必须支持 `get_endpoint(port)` 取 execd endpoint。
    httpx_client: 共享 client（连接池复用）；调用方负责 close。

    读超时不限（和 SDK 的 SSE client 一致）；阻塞时长受 execd 本身处理时间决定。
    """
    # 懒拿常量，避免 opensandbox 未装时 import 失败（按项目 optional-dep 约定）
    constants = importlib.import_module("opensandbox.constants")
    execd_port = constants.DEFAULT_EXECD_PORT

    endpoint = await sandbox.get_endpoint(execd_port)
    url = f"http://{endpoint.endpoint}{_RUN_COMMAND_PATH}"

    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        **getattr(endpoint, "headers", {}),
    }

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    exit_code: int | None = None
    error_text: str | None = None

    # 单次调用的 read 超时用 None（SSE 靠 server 推）；connect/write 受控。
    request_timeout = httpx.Timeout(
        connect=connect_timeout, read=None, write=connect_timeout, pool=connect_timeout
    )

    async with httpx_client.stream(
        "POST", url,
        json={"command": command},
        headers=headers,
        timeout=request_timeout,
    ) as response:
        if response.status_code != 200:
            await response.aread()
            body = response.text
            return FastExecResult(
                exit_code=None, stdout="", stderr="",
                error=f"HTTP {response.status_code}: {body[:500]}",
            )

        async for line in response.aiter_lines():
            line = line.strip()
            if not line or line.startswith((":", "event:", "id:", "retry:")):
                continue
            # SDK 的 SSE 帧是裸 JSON 一行，不带 `data:` 前缀（已验证）。
            # 保险起见也兼容 `data: {...}` 格式。
            if line.startswith("data:"):
                line = line[5:].strip()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            text = event.get("text", "") or ""

            if etype == "stdout":
                stdout_parts.append(text)
            elif etype == "stderr":
                stderr_parts.append(text)
            elif etype == "execution_complete":
                # 成功路径：execd 发出 execution_complete → exit_code=0
                # （非 0 退出由 error 事件分支处理，见下）
                exit_code = 0
                break  # 早退，避免等 server 再挂 1s
            elif etype == "error":
                # execd error 事件的真实结构（实测 v1.0.13）：
                #   {"type":"error","error":{"ename":"CommandExecError",
                #                            "evalue":"<exit>",
                #                            "traceback":["exit status <exit>"]}}
                err_obj = event.get("error") or {}
                raw = err_obj.get("evalue") if isinstance(err_obj, dict) else None
                if raw is None:
                    # 旧 execd 或其他形态兜底
                    raw = event.get("value") or event.get("text")
                error_text = str(raw) if raw is not None else "command failed"
                try:
                    exit_code = int(raw) if raw is not None else None
                except (TypeError, ValueError):
                    exit_code = None
                break

    return FastExecResult(
        exit_code=exit_code,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        error=error_text,
    )
