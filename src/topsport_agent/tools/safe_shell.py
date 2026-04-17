from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class ShellInjectionError(Exception):
    pass


_SHELL_INTERPRETERS = {"sh", "bash", "zsh", "fish", "dash", "csh", "tcsh", "ksh"}


async def safe_exec(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_output: int = 15_000,
    allowed_commands: set[str] | None = None,
) -> dict[str, Any]:
    """只允许 execFile 模式（argv 列表），拒绝一切 shell=True 路径和 shell -c 变体，从根源阻断注入。"""
    if not isinstance(command, list):
        raise ShellInjectionError(
            "command must be a list of strings [program, arg1, arg2, ...]; "
            "string commands are rejected to prevent shell injection"
        )
    if not command:
        raise ValueError("command list must not be empty")
    for i, part in enumerate(command):
        if not isinstance(part, str):
            raise ShellInjectionError(
                f"command[{i}] must be a string, got {type(part).__name__}"
            )
    # 拦截 "bash -c '...'" 这类绕过 argv 隔离的写法，防止 LLM 构造注入向量。
    executable = Path(command[0]).name.lower()
    if executable in _SHELL_INTERPRETERS and any(
        flag in command[1:] for flag in ("-c", "-lc", "--login", "-ic")
    ):
        raise ShellInjectionError(
            f"shell interpreter '{command[0]}' with -c flag is rejected; "
            "pass the actual command as argv elements"
        )
    if allowed_commands is not None and executable not in allowed_commands:
        raise ShellInjectionError(
            f"command '{command[0]}' not in allowed_commands: {sorted(allowed_commands)}"
        )

    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        env=env,
    )

    # 字节上限 = 字符上限 x 4，覆盖最坏 UTF-8 编码情况，防止 OOM。
    byte_limit = max_output * 4

    try:
        stdout_bytes, stdout_trunc = await asyncio.wait_for(
            _read_limited(proc.stdout, byte_limit), timeout=timeout
        )
        stderr_bytes, stderr_trunc = await asyncio.wait_for(
            _read_limited(proc.stderr, byte_limit), timeout=timeout
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"timed out after {timeout}s",
            "timed_out": True,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

    # 主输出已读完，给进程最后 5 秒退出；超时则强杀，避免僵尸进程。
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    return {
        "exit_code": proc.returncode or 0,
        "stdout": stdout_text[:max_output],
        "stderr": stderr_text[:max_output],
        "timed_out": False,
        "stdout_truncated": stdout_trunc or len(stdout_text) > max_output,
        "stderr_truncated": stderr_trunc or len(stderr_text) > max_output,
    }


async def _read_limited(
    stream: asyncio.StreamReader | None, limit: int
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    chunks: list[bytes] = []
    total = 0
    while total < limit:
        chunk = await stream.read(min(8192, limit - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    truncated = total >= limit
    return b"".join(chunks), truncated
