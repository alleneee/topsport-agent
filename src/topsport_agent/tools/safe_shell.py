from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class ShellInjectionError(Exception):
    pass


async def safe_exec(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_output: int = 15_000,
) -> dict[str, Any]:
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

    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        env=env,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
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

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    return {
        "exit_code": proc.returncode,
        "stdout": stdout_text[:max_output],
        "stderr": stderr_text[:max_output],
        "timed_out": False,
        "stdout_truncated": len(stdout_text) > max_output,
        "stderr_truncated": len(stderr_text) > max_output,
    }
