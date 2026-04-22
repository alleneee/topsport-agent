"""OpenSandbox 工具 source：向 Engine 暴露 shell / read_file / write_file。

每个 handler 从 ctx.session_id 向 pool 拿对应 sandbox（首次 tool 调用触发创建）。
不处理取消信号（ctx.cancel_event）—— 探索阶段依赖 sandbox 本身的 timeout
以及 Engine 对 handler 整体的中断兜底。
"""
from __future__ import annotations

import importlib
from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .fast_exec import fast_run_command
from .pool import OpenSandboxPool

_OUTPUT_CAP = 15_000


def _join_output(logs: Any, field: str) -> str:
    """从 ExecutionLogs.{stdout,stderr} (list[OutputMessage]) 聚合 text。

    logs 或字段缺失 / 非列表时返回空串，调用方据此 fallback。
    """
    if logs is None:
        return ""
    seq = getattr(logs, field, None)
    if not isinstance(seq, list):
        return ""
    return "".join((getattr(msg, "text", "") or "") for msg in seq)


class OpenSandboxToolSource:
    name = "opensandbox"

    def __init__(
        self,
        pool: OpenSandboxPool,
        *,
        prefix: str = "sandbox",
        write_entry_cls: Any | None = None,
        fast_shell: bool = True,
    ) -> None:
        """
        write_entry_cls: 可注入 mock 类避免测试硬依赖 opensandbox。
        fast_shell: 默认启用 fast_exec（绕过 SDK 的 1s SSE 阻塞）；
                    测试传 False 退回 SDK 路径（sandbox.commands.run）以复用 MockSandbox。
        """
        self._pool = pool
        self._prefix = prefix
        self._write_entry_cls = write_entry_cls
        self._fast_shell = fast_shell

    async def list_tools(self) -> list[ToolSpec]:
        return [
            self._shell_spec(),
            self._read_file_spec(),
            self._write_file_spec(),
        ]

    def _shell_spec(self) -> ToolSpec:
        return ToolSpec(
            name=f"{self._prefix}_shell",
            description=(
                "Execute a shell command inside the session's isolated sandbox. "
                "Returns stdout/stderr truncated at 15000 chars."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                },
                "required": ["command"],
            },
            handler=self._shell_handler,
        )

    def _read_file_spec(self) -> ToolSpec:
        return ToolSpec(
            name=f"{self._prefix}_read_file",
            description=(
                "Read a text file inside the session's sandbox. "
                "Path must be absolute."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path inside the sandbox",
                    },
                },
                "required": ["path"],
            },
            handler=self._read_file_handler,
        )

    def _write_file_spec(self) -> ToolSpec:
        return ToolSpec(
            name=f"{self._prefix}_write_file",
            description=(
                "Write content to a file inside the session's sandbox "
                "(creates or overwrites). Path must be absolute."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path inside the sandbox",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content",
                    },
                },
                "required": ["path", "content"],
            },
            handler=self._write_file_handler,
        )

    async def _shell_handler(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return {"ok": False, "error": "command must be a non-empty string"}
        sandbox = await self._pool.acquire(ctx.session_id)
        if self._fast_shell:
            return await self._shell_fast(sandbox, command)
        return await self._shell_via_sdk(sandbox, command)

    async def _shell_fast(self, sandbox: Any, command: str) -> dict[str, Any]:
        """走 fast_exec：跳过 SDK 的 SSE 关流阻塞，延迟从 ~1s 降到 ~15ms。"""
        try:
            http_client = await self._pool.get_http_client()
            result = await fast_run_command(sandbox, command, httpx_client=http_client)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        exit_code = result.exit_code
        return {
            "ok": exit_code == 0 and result.error is None,
            "exit_code": exit_code,
            "stdout": result.stdout[:_OUTPUT_CAP],
            "stderr": result.stderr[:_OUTPUT_CAP],
            "error": result.error,
        }

    async def _shell_via_sdk(self, sandbox: Any, command: str) -> dict[str, Any]:
        """SDK 路径：走 sandbox.commands.run。保留用于测试 mock 兼容。"""
        try:
            execution = await sandbox.commands.run(command)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        stdout = _join_output(getattr(execution, "logs", None), "stdout") or (
            str(getattr(execution, "stdout", "") or "")
        )
        stderr = _join_output(getattr(execution, "logs", None), "stderr") or (
            str(getattr(execution, "stderr", "") or "")
        )
        exit_code = getattr(execution, "exit_code", None)
        err_obj = getattr(execution, "error", None)
        return {
            "ok": exit_code == 0 and err_obj is None,
            "exit_code": exit_code,
            "stdout": stdout[:_OUTPUT_CAP],
            "stderr": stderr[:_OUTPUT_CAP],
            "error": (str(err_obj) if err_obj is not None else None),
        }

    async def _read_file_handler(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        path = args.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            return {"ok": False, "error": "path must be an absolute string"}
        sandbox = await self._pool.acquire(ctx.session_id)
        try:
            content = await sandbox.files.read_file(path)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "path": path, "content": str(content)}

    async def _write_file_handler(
        self, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path.startswith("/"):
            return {"ok": False, "error": "path must be an absolute string"}
        if not isinstance(content, str):
            return {"ok": False, "error": "content must be a string"}
        sandbox = await self._pool.acquire(ctx.session_id)
        entry_cls = self._resolve_write_entry_cls()
        try:
            await sandbox.files.write_files([entry_cls(path=path, data=content)])
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "path": path,
            "bytes_written": len(content.encode("utf-8")),
        }

    def _resolve_write_entry_cls(self) -> Any:
        if self._write_entry_cls is None:
            # opensandbox>=0.1.7 的 WriteEntry 位于 models.filesystem 子模块，
            # 不在顶层 opensandbox.* 命名空间。
            mod_name = "opensandbox.models.filesystem"
            filesystem_mod = importlib.import_module(mod_name)
            self._write_entry_cls = filesystem_mod.WriteEntry
        return self._write_entry_cls
