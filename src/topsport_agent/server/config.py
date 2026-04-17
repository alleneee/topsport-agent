"""服务配置：从环境变量读取，启动 Agent 所需的 provider 凭证与默认模型。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ServerConfig:
    """HTTP 服务启动配置。

    provider/base_url/api_key 全局一份，session 级别只定制 model/system_prompt。
    鉴权 secure by default：auth_required=True；未设 auth_token / auth_tokens_file
    时启动失败，避免无意中暴露未鉴权接口。
    """

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""
    base_url: str | None = None
    default_model: str = ""
    session_ttl_seconds: int = 3600
    max_sessions: int = 128

    # 鉴权
    auth_required: bool = True
    auth_token: str = ""  # 单 token 简易模式，principal 恒为 "default"
    auth_tokens_file: str = ""  # 多租户：JSON {token: principal}
    # 对外服务的 Agent 能力闸门 —— 默认全关，避免 CR-01 默认暴露文件/插件能力
    enable_file_tools: bool = False
    enable_skills: bool = False
    enable_plugins: bool = False
    # Plan 执行的硬上限，防止客户端提交超大 max_steps 绕过运营预算
    max_plan_steps: int = 20

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
            auth_required=_parse_bool(os.environ.get("AUTH_REQUIRED"), default=True),
            auth_token=os.environ.get("AUTH_TOKEN", ""),
            auth_tokens_file=os.environ.get("AUTH_TOKENS_FILE", ""),
            enable_file_tools=_parse_bool(
                os.environ.get("ENABLE_FILE_TOOLS"), default=False
            ),
            enable_skills=_parse_bool(os.environ.get("ENABLE_SKILLS"), default=False),
            enable_plugins=_parse_bool(os.environ.get("ENABLE_PLUGINS"), default=False),
            max_plan_steps=int(os.environ.get("MAX_PLAN_STEPS", "20")),
        )


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
