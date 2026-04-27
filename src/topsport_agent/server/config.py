"""服务配置：从环境变量读取，启动 Agent 所需的 provider 凭证与默认模型。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class ServerConfig:
    """HTTP 服务启动配置。

    provider/base_url/api_key 全局一份，session 级别只定制 model/system_prompt。
    鉴权 secure by default：auth_required=True；未设 auth_token / auth_tokens_file
    时启动失败，避免无意中暴露未鉴权接口。
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # api_key 是 LLM provider 凭证；repr=False 防止启动 log / Langfuse trace
    # / FastAPI debug traceback 把它打到日志或前端。
    api_key: str = field(default="", repr=False)
    base_url: str | None = None
    default_model: str = ""
    session_ttl_seconds: int = 3600
    max_sessions: int = 128

    # 鉴权
    auth_required: bool = True
    auth_token: str = field(default="", repr=False)  # 单 token 简易模式，principal 恒为 "default"
    auth_tokens_file: str = ""  # 多租户：JSON {token: principal}
    # 对外服务的 Agent 能力闸门 —— 默认全关，避免 CR-01 默认暴露文件/插件能力。
    # enable_memory 从 default_agent 的硬编码 True 迁移过来：server 部署默认关，
    # 因为 memory 会把上下文写入 FileMemoryStore（host 文件系统），多租户场景
    # 必须通过 persona+assignment 明确授权（memory.write 能力）才打开。
    enable_file_tools: bool = False
    enable_skills: bool = False
    enable_memory: bool = False
    enable_plugins: bool = False
    enable_browser: bool = False
    # Image generation（OpenAI /v1/images/generations）。客户端和 chat 的 API_KEY / BASE_URL
    # 共用，`image_gen_model` 作为请求缺省模型；IMAGE_GEN_BASE_URL 单独指定可覆盖 chat 的
    enable_image_gen: bool = False
    image_gen_model: str = ""
    image_gen_base_url: str | None = None
    # Langfuse tracing（可观测性）。enable=true 且 LANGFUSE_PUBLIC_KEY/SECRET_KEY 齐
    # 才真正构造；缺 key 时启动失败（fail-fast，避免运维以为开了实际没数据）。
    enable_langfuse: bool = False
    # MCP（Model Context Protocol）tool sources。指向 claude-desktop 兼容的 JSON 配置；
    # 启动时加载，server 端对所有 session 生效。当前不支持 per-tenant 不同 MCP。
    mcp_config_path: str | None = None
    # Built-in Brave Search MCP server（@brave/brave-search-mcp-server via npx）。
    # enable_brave_search=True 且 brave_api_key 非空时，server 启动期自动注册到
    # MCPManager；与 mcp_config_path 共存（同一 MCPManager 同时持有）。
    # API key 不要硬编码到源码 / git；从 BRAVE_API_KEY env 读取。
    enable_brave_search: bool = False
    # repr=False：防止 logger.info("starting cfg=%s", cfg) / debug traceback
    # / Langfuse 配置 dump 把 API key 暴露到日志或前端。
    brave_api_key: str = field(default="", repr=False)
    # 通过 MCP `roots` capability 暴露给所有 MCP server 的文件系统根（绝对路径）。
    # 空表示不声明 roots 能力（向后兼容）。每个路径在启动期 resolve 到 file:// URI；
    # 不存在的路径仍会注册（server 端自行处理），但相对路径会按当前 cwd 展开
    # —— 推荐都用绝对路径写到 env / config，避免 deployment 漂移。
    # 视为只读：frozen=True 只阻止 rebind，list 自身仍可 mutate；不要在运行期
    # `cfg.mcp_roots.append(...)`，会让 _build_mcp_manager 二次调用看到污染快照。
    mcp_roots: list[str] = field(default_factory=list)
    # MCP 服务器日志级别。空字符串或 "off" / "none" → 不订阅日志（向后兼容）。
    # 合法值: debug / info / notice / warning / error / critical / alert / emergency
    # （MCP spec 8 级 syslog form）。设置后 client 会在每次 initialize 后调
    # `logging/setLevel(level)`，并把 server 推来的 notifications/message 路由到
    # `topsport_agent.mcp.server.<client_name>` 日志器。
    mcp_log_level: str = ""
    # 启用 MCP `progress` 通知消费：每次 call_tool 自动带上 progress_callback，
    # server 推来的 notifications/progress 流入 `topsport_agent.mcp.progress.<client>`
    # 日志器。默认关闭——长时工具的进度可观测性是 opt-in（额外日志量）。
    #
    # 与 mcp_log_level 的关系：两个 capability 正交。progress 走
    # `topsport_agent.mcp.progress.<client>`，server 普通 log 走
    # `topsport_agent.mcp.server.<client>`。某些 server 把进度写成
    # logging notification 而非 progress notification，此时两者都开会让
    # 同一进度信息出现在两个日志树（运营侧自行去重）。
    enable_mcp_progress: bool = False
    # Per-session disk workspace base directory. Each session gets
    # <workspace_root>/<safe_session_id>/files/ as its file_ops sandbox.
    # Empty string / None 默认回落到 ~/.topsport-agent/workspaces/。
    # 设为某绝对路径可放到专用存储卷（建议企业部署单独挂盘）。
    workspace_root: str | None = None
    # session 关闭时是否删除对应 workspace 目录。False（默认）保留便于 debug /
    # retention policy 由外部清理；True 适合严格多租户隔离场景。
    workspace_delete_on_close: bool = False
    # Plan 执行的硬上限，防止客户端提交超大 max_steps 绕过运营预算
    max_plan_steps: int = 20
    # Chat 路径的 Engine 每会话最大 ReAct 步数（工具调用+LLM 往返次数上限）
    max_chat_steps: int = 20
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

    # Database (pluggable skeleton; see database/)
    enable_database: bool = False
    database_backend: str = "postgres"          # only applied when enable_database=True
    database_url: str | None = None
    database_pool_min: int = 1
    database_pool_max: int = 10
    database_timeout_seconds: float = 30.0

    # Rate limiting (Redis-backed)
    enable_rate_limit: bool = False
    ratelimit_redis_url: str | None = None
    ratelimit_window_seconds: int = 60
    ratelimit_per_ip: int = 300          # 0 = disable this dimension
    ratelimit_per_principal: int = 60
    ratelimit_per_tenant: int = 1000
    ratelimit_per_route_default: int = 0
    ratelimit_routes: dict[str, int] = field(default_factory=dict)
    ratelimit_trust_forwarded_for: bool = False
    ratelimit_fail_open: bool = True

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
            enable_memory=_parse_bool(os.environ.get("ENABLE_MEMORY"), default=False),
            enable_plugins=_parse_bool(os.environ.get("ENABLE_PLUGINS"), default=False),
            enable_browser=_parse_bool(os.environ.get("ENABLE_BROWSER"), default=False),
            enable_image_gen=_parse_bool(os.environ.get("ENABLE_IMAGE_GEN"), default=False),
            image_gen_model=os.environ.get("IMAGE_GEN_MODEL", "") or "",
            image_gen_base_url=os.environ.get("IMAGE_GEN_BASE_URL") or None,
            enable_langfuse=_parse_bool(os.environ.get("ENABLE_LANGFUSE"), default=False),
            mcp_config_path=os.environ.get("MCP_CONFIG_PATH") or None,
            enable_brave_search=_parse_bool(
                os.environ.get("ENABLE_BRAVE_SEARCH"), default=False
            ),
            brave_api_key=os.environ.get("BRAVE_API_KEY", ""),
            mcp_roots=_parse_path_list(os.environ.get("MCP_ROOTS")),
            mcp_log_level=os.environ.get("MCP_LOG_LEVEL", "").strip().lower(),
            enable_mcp_progress=_parse_bool(
                os.environ.get("ENABLE_MCP_PROGRESS"), default=False
            ),
            workspace_root=os.environ.get("WORKSPACE_ROOT") or None,
            workspace_delete_on_close=_parse_bool(
                os.environ.get("WORKSPACE_DELETE_ON_CLOSE"), default=False
            ),
            prompt_injection_guard=_parse_bool(
                os.environ.get("PROMPT_INJECTION_GUARD"), default=True
            ),
            log_format=os.environ.get("LOG_FORMAT", "text").strip().lower() or "text",
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            max_plan_steps=int(os.environ.get("MAX_PLAN_STEPS", "20")),
            max_chat_steps=int(os.environ.get("MAX_CHAT_STEPS", "20")),
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
            enable_database=_parse_bool(
                os.environ.get("ENABLE_DATABASE"), default=False
            ),
            database_backend=os.environ.get("DATABASE_BACKEND", "postgres"),
            database_url=os.environ.get("DATABASE_URL") or None,
            database_pool_min=int(os.environ.get("DATABASE_POOL_MIN", "1")),
            database_pool_max=int(os.environ.get("DATABASE_POOL_MAX", "10")),
            database_timeout_seconds=float(
                os.environ.get("DATABASE_TIMEOUT_SECONDS", "30")
            ),
            enable_rate_limit=_parse_bool(
                os.environ.get("ENABLE_RATE_LIMIT"), default=False
            ),
            ratelimit_redis_url=os.environ.get("RATELIMIT_REDIS_URL") or None,
            ratelimit_window_seconds=int(
                os.environ.get("RATELIMIT_WINDOW_SECONDS", "60")
            ),
            ratelimit_per_ip=int(os.environ.get("RATELIMIT_PER_IP", "300")),
            ratelimit_per_principal=int(
                os.environ.get("RATELIMIT_PER_PRINCIPAL", "60")
            ),
            ratelimit_per_tenant=int(
                os.environ.get("RATELIMIT_PER_TENANT", "1000")
            ),
            ratelimit_per_route_default=int(
                os.environ.get("RATELIMIT_PER_ROUTE_DEFAULT", "0")
            ),
            ratelimit_routes=_parse_route_limits(
                os.environ.get("RATELIMIT_ROUTES")
            ),
            ratelimit_trust_forwarded_for=_parse_bool(
                os.environ.get("RATELIMIT_TRUST_FORWARDED_FOR"), default=False
            ),
            ratelimit_fail_open=_parse_bool(
                os.environ.get("RATELIMIT_FAIL_OPEN"), default=True
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


def _parse_path_list(raw: str | None) -> list[str]:
    """`MCP_ROOTS=/a:/b:/c` 风格分隔（os.pathsep — Unix `:` / Windows `;`）。
    空字符串 / 未设置 → 空列表。

    Windows 注意：路径含驱动盘符（`C:\\proj`）时不要用 `:` 分隔；用
    Windows 默认的 `;`，或者把每个 root 单独配置。
    """
    if raw is None or not raw.strip():
        return []
    return [seg.strip() for seg in raw.split(os.pathsep) if seg.strip()]


def _parse_route_limits(raw: str | None) -> dict[str, int]:
    """Parse RATELIMIT_ROUTES JSON env. Empty/unset → empty dict.

    Raises ValueError on malformed JSON (fail-fast at startup).
    """
    if raw is None or not raw.strip():
        return {}
    import json

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"RATELIMIT_ROUTES must be a JSON object, got: {type(data).__name__}"
        )
    return {str(k): int(v) for k, v in data.items()}
