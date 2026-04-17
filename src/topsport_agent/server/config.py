"""服务配置：从环境变量读取，启动 Agent 所需的 provider 凭证与默认模型。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ServerConfig:
    """HTTP 服务启动配置。

    provider/base_url/api_key 全局一份，session 级别只定制 model/system_prompt。
    """

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""
    base_url: str | None = None
    default_model: str = ""
    session_ttl_seconds: int = 3600
    max_sessions: int = 128

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8000")),
            api_key=os.environ.get("API_KEY", ""),
            base_url=os.environ.get("BASE_URL"),
            default_model=os.environ.get("MODEL", ""),
            session_ttl_seconds=int(os.environ.get("SESSION_TTL_SECONDS", "3600")),
            max_sessions=int(os.environ.get("MAX_SESSIONS", "128")),
        )
