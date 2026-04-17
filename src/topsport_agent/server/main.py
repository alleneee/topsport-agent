"""`topsport-agent-serve` 入口：加载 .env，启动 uvicorn 跑 FastAPI app。

H-S3: .env 文件不再默认从 CWD 自动加载。只有通过 `--env-file PATH` 显式指定的
路径才会读取，且必须通过属主与权限校验（属主 == 当前 uid，mode & 0o077 == 0）。
这避免了 `cd` 进恶意目录时 API_KEY / BASE_URL / LANGFUSE_* 被静默劫持。
"""

from __future__ import annotations

import argparse
import importlib
import os
import stat
import sys
from pathlib import Path

from .app import create_app
from .config import ServerConfig


class _DotenvRefused(Exception):
    """属主或权限不合规时抛出，提示调用方修权限后重试。"""


def _validate_dotenv_permissions(path: Path) -> None:
    """拒绝:
    - 不属于当前 uid 的文件（防止他人放置恶意 .env）
    - group/other 任意读写位置位（mode & 0o077 != 0）
    仅在 POSIX 有意义；Windows 跳过权限检查但仍要求文件存在。
    """
    if os.name != "posix":
        return
    st = path.stat()
    current_uid = os.geteuid()
    if st.st_uid != current_uid:
        raise _DotenvRefused(
            f"env-file {path} is owned by uid={st.st_uid}, not current uid={current_uid}"
        )
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o077:
        raise _DotenvRefused(
            f"env-file {path} has permissive mode {oct(mode)}; "
            "chmod 600 before loading"
        )


def _load_dotenv(path: Path) -> None:
    """解析 KEY=VALUE 格式，只在 key 未在 env 里时注入；属主/权限不合规直接抛。"""
    _validate_dotenv_permissions(path)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="topsport-agent-serve",
        description="Run topsport-agent HTTP + SSE server (OpenAI-compatible).",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "openai"],
        help="provider 身份 (默认从 MODEL 前缀推导)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="路径必须属于当前用户且 mode=0600；不再默认从 CWD/.env 加载",
    )
    args = parser.parse_args()

    if args.env_file:
        env_path = Path(args.env_file).expanduser().resolve()
        if not env_path.is_file():
            print(f"env-file not found: {env_path}", file=sys.stderr)
            sys.exit(1)
        try:
            _load_dotenv(env_path)
        except _DotenvRefused as exc:
            print(f"refusing to load env-file: {exc}", file=sys.stderr)
            sys.exit(1)

    cfg = ServerConfig.from_env()
    if args.host:
        cfg = ServerConfig(**{**cfg.__dict__, "host": args.host})
    if args.port:
        cfg = ServerConfig(**{**cfg.__dict__, "port": args.port})

    provider_name = args.provider
    if not provider_name:
        model = os.environ.get("MODEL", "")
        provider_name = model.partition("/")[0].strip().lower() or "anthropic"
        if provider_name not in ("anthropic", "openai"):
            print(f"invalid provider derived from MODEL: {provider_name!r}", file=sys.stderr)
            sys.exit(1)

    if not cfg.api_key:
        print("API_KEY is required (set in environment or via --env-file)", file=sys.stderr)
        sys.exit(1)

    app = create_app(cfg, provider_name=provider_name)

    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
