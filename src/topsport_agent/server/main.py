"""`topsport-agent-serve` 入口：加载 .env，启动 uvicorn 跑 FastAPI app。"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from .app import create_app
from .config import ServerConfig


def _load_dotenv() -> None:
    env = Path.cwd() / ".env"
    if not env.is_file():
        return
    for raw in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


def main() -> None:
    _load_dotenv()

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
    args = parser.parse_args()

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
        print("API_KEY is required (set in .env or environment)", file=sys.stderr)
        sys.exit(1)

    app = create_app(cfg, provider_name=provider_name)

    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
