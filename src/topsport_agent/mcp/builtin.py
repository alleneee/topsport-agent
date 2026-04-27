"""Built-in MCP server configurations.

Convenience factories for popular MCP servers, so operators can wire them
into an Agent without writing full JSON configs by hand. Each factory
returns a ready-to-register `MCPServerConfig`; pass the result to
`MCPManager.register(MCPClient.from_config(cfg))` or include it in a
`MCPManager` instance built from environment.

Currently supported:
    - Brave Search (`@brave/brave-search-mcp-server`, npm) — web search +
      image search + news search via the Brave Search API. Bridged into
      the Agent's tool surface as `brave-search.brave_web_search` /
      `brave-search.brave_image_search` / etc. (each MCP tool gets the
      `<server-name>.<tool-name>` prefix from MCPToolSource).

The `--transport http` flag commonly seen in Brave's README is meant for
running the server as a standalone HTTP listener; topsport-agent uses
the SDK's stdio transport here so the npm subprocess wires directly to
the agent over stdin/stdout — no HTTP port allocation, no port-conflict
surprises in multi-tenant deployments.
"""

from __future__ import annotations

import logging
import shutil

from .types import MCPServerConfig, MCPTransport

_logger = logging.getLogger(__name__)

BRAVE_DEFAULT_NAME = "brave-search"
BRAVE_PACKAGE = "@brave/brave-search-mcp-server"


def brave_search_config(
    api_key: str,
    *,
    name: str = BRAVE_DEFAULT_NAME,
    package: str = BRAVE_PACKAGE,
    permissions: frozenset[str] = frozenset(),
    cache_ttl: float | None = 60.0,
    timeout: float = 30.0,
    extra_env: dict[str, str] | None = None,
) -> MCPServerConfig:
    """Build a stdio MCPServerConfig for the Brave Search MCP server.

    Equivalent to the npm command:
        npx -y @brave/brave-search-mcp-server

    With BRAVE_API_KEY env populated. The bridged tools appear in the
    Agent's pool as `<name>.brave_web_search` etc. Adjust `name` to scope
    the prefix; `permissions` to gate visibility behind capability ACLs.

    Empty `api_key` raises ValueError to fail fast — Brave Search refuses
    every request without an API key, and silent registration would leave
    every web_search tool call returning a cryptic 401 to the LLM.
    """
    if not api_key.strip():
        raise ValueError(
            "brave_search_config: api_key is required (get one at "
            "https://api.search.brave.com/app/dashboard)"
        )
    # Preflight: npx 不在 PATH 时，错误延迟到首次 list_tools 才暴露，运维难定位。
    # 这里只 warn，不 raise — CI / 容器环境可能在启动后才挂载 node_modules。
    if shutil.which(command := "npx") is None:
        _logger.warning(
            "brave_search_config: %r not found in PATH; first MCP call will fail "
            "until node/npm is available. Install with `npm install -g npm` or "
            "ensure your container image includes node>=18.",
            command,
        )
    # api_key 是真理来源：extra_env 不能覆盖 BRAVE_API_KEY（防止 caller 误传
    # extra_env={"BRAVE_API_KEY": "..."} 而绕过 api_key 显式参数的安全审计）。
    env: dict[str, str] = dict(extra_env or {})
    env["BRAVE_API_KEY"] = api_key
    return MCPServerConfig(
        name=name,
        transport=MCPTransport.STDIO,
        command="npx",
        args=["-y", package],
        env=env,
        permissions=permissions,
        cache_ttl=cache_ttl,
        timeout=timeout,
    )


__all__ = [
    "BRAVE_DEFAULT_NAME",
    "BRAVE_PACKAGE",
    "brave_search_config",
]
