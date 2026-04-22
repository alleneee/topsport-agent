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
    # 进程收到 SIGTERM 后等待 in-flight 请求的最大秒数（H-R5 graceful drain）
    drain_timeout_seconds: float = 25.0

    # Prompt injection 防御：对 untrusted 工具结果（browser / MCP）做消毒，
    # 并在 system prompt 注入 security guard 告知 LLM 围栏语义。默认开启。
    prompt_injection_guard: bool = True
    # 日志格式：text（stdlib 默认）或 json（结构化，便于 ELK/Loki 对接）。
    log_format: str = "text"
    log_level: str = "INFO"

    # OpenSandbox 集成（可选）
    # 启用时：tool_source 注入 sandbox_shell/read_file/write_file；
    # 启用时自动禁用本地 file_ops（避免 SEC-001：LLM 越过沙箱读宿主文件系统）
    sandbox_enabled: bool = False
    sandbox_domain: str = "localhost:8090"
    sandbox_image: str = "ubuntu"
    sandbox_per_tenant_max: int | None = None  # None = 不限
    sandbox_per_tenant_timeout_seconds: float | None = None  # None = 阻塞到有空位
    sandbox_idle_pause_seconds: float | None = 300.0  # None = 禁用 idle pause
    sandbox_use_server_proxy: bool = True  # 对 Docker bridge 部署必须 True

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
            prompt_injection_guard=_parse_bool(
                os.environ.get("PROMPT_INJECTION_GUARD"), default=True
            ),
            log_format=os.environ.get("LOG_FORMAT", "text").strip().lower() or "text",
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            max_plan_steps=int(os.environ.get("MAX_PLAN_STEPS", "20")),
            drain_timeout_seconds=float(
                os.environ.get("DRAIN_TIMEOUT_SECONDS", "25")
            ),
            sandbox_enabled=_parse_bool(
                os.environ.get("SANDBOX_ENABLED"), default=False
            ),
            sandbox_domain=os.environ.get("SANDBOX_DOMAIN", "localhost:8090"),
            sandbox_image=os.environ.get("SANDBOX_IMAGE", "ubuntu"),
            sandbox_per_tenant_max=_parse_optional_int(
                os.environ.get("SANDBOX_PER_TENANT_MAX")
            ),
            sandbox_per_tenant_timeout_seconds=_parse_optional_float(
                os.environ.get("SANDBOX_PER_TENANT_TIMEOUT_SECONDS")
            ),
            sandbox_idle_pause_seconds=_parse_optional_float(
                os.environ.get("SANDBOX_IDLE_PAUSE_SECONDS"), default=300.0
            ),
            sandbox_use_server_proxy=_parse_bool(
                os.environ.get("SANDBOX_USE_SERVER_PROXY"), default=True
            ),
        )


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _parse_optional_float(raw: str | None, *, default: float | None = None) -> float | None:
    if raw is None or not raw.strip():
        return default
    s = raw.strip().lower()
    if s in {"none", "null", "off", "false", "disable", "disabled"}:
        return None
    return float(raw)
