"""Built-in Brave Search MCP server integration tests.

覆盖：
- brave_search_config 默认 + 自定义 name / extra_env / cache_ttl 透传
- 空 api_key 触发 ValueError fail-fast
- ServerConfig 字段从 env 读取
- _build_mcp_manager 在 enable_brave_search 路径上注册 brave，缺 key 失败
- enable_brave_search + mcp_config_path 同时启用：合并到同一 MCPManager
- 同名冲突（mcp_config_path 已注册 "brave-search"）→ fail-fast

不真正起 npx 子进程；只验证 MCPServerConfig 字段 + manager 注册结构。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.mcp import brave_search_config
from topsport_agent.mcp.types import MCPTransport
from topsport_agent.server.app import _build_mcp_manager
from topsport_agent.server.config import ServerConfig


# ---------------------------------------------------------------------------
# brave_search_config factory
# ---------------------------------------------------------------------------


def test_brave_search_config_defaults_to_stdio_npx() -> None:
    cfg = brave_search_config(api_key="test-key")

    assert cfg.name == "brave-search"
    assert cfg.transport == MCPTransport.STDIO
    assert cfg.command == "npx"
    assert cfg.args == ["-y", "@brave/brave-search-mcp-server"]
    assert cfg.env == {"BRAVE_API_KEY": "test-key"}
    assert cfg.cache_ttl == 60.0
    assert cfg.timeout == 30.0


def test_brave_search_config_custom_name_and_env() -> None:
    cfg = brave_search_config(
        api_key="k",
        name="brave-prod",
        extra_env={"DEBUG": "1"},
        cache_ttl=300.0,
        timeout=10.0,
        permissions=frozenset({"web.search"}),
    )

    assert cfg.name == "brave-prod"
    assert cfg.env == {"BRAVE_API_KEY": "k", "DEBUG": "1"}
    assert cfg.cache_ttl == 300.0
    assert cfg.timeout == 10.0
    assert cfg.permissions == frozenset({"web.search"})


def test_brave_search_config_empty_api_key_raises() -> None:
    with pytest.raises(ValueError, match="api_key is required"):
        brave_search_config(api_key="")
    with pytest.raises(ValueError, match="api_key is required"):
        brave_search_config(api_key="   ")


def test_brave_search_config_cache_ttl_none_propagates() -> None:
    """cache_ttl=None（永不过期）必须透传到生成的 MCPServerConfig。"""
    cfg = brave_search_config(api_key="k", cache_ttl=None)
    assert cfg.cache_ttl is None


def test_brave_search_config_warns_when_npx_missing(monkeypatch, caplog) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with caplog.at_level("WARNING", logger="topsport_agent.mcp.builtin"):
        brave_search_config(api_key="k")
    assert any("not found in PATH" in rec.message for rec in caplog.records)


def test_brave_search_config_extra_env_does_not_override_api_key() -> None:
    """API key 是 factory 入参的真理来源；extra_env 不应能覆盖它。"""
    cfg = brave_search_config(
        api_key="primary",
        extra_env={"BRAVE_API_KEY": "evil-override"},
    )
    # extra_env 应用在 BRAVE_API_KEY 之后才会被覆盖；当前实现 BRAVE_API_KEY 先
    # 写入再 update，extra_env 会覆盖。这是一条契约：本测试钉死期望行为。
    # 决定：让 api_key 参数赢（更安全），后续若改实现这条 test 会守住。
    assert cfg.env["BRAVE_API_KEY"] == "primary", (
        "api_key parameter must remain the source of truth"
    )


# ---------------------------------------------------------------------------
# ServerConfig env wiring
# ---------------------------------------------------------------------------


def test_server_config_default_disables_brave_search() -> None:
    cfg = ServerConfig()
    assert cfg.enable_brave_search is False
    assert cfg.brave_api_key == ""


def test_server_config_from_env_reads_brave_fields(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_BRAVE_SEARCH", "true")
    monkeypatch.setenv("BRAVE_API_KEY", "env-key")
    cfg = ServerConfig.from_env()
    assert cfg.enable_brave_search is True
    assert cfg.brave_api_key == "env-key"


def test_server_config_from_env_brave_key_unset_defaults_to_empty(
    monkeypatch,
) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_BRAVE_SEARCH", raising=False)
    cfg = ServerConfig.from_env()
    assert cfg.brave_api_key == ""
    assert cfg.enable_brave_search is False


def test_brave_api_key_not_in_server_config_repr() -> None:
    """API key 必须不出现在 ServerConfig.__repr__ 输出里（避免日志泄漏）。"""
    cfg = ServerConfig(
        enable_brave_search=True,
        brave_api_key="secret-do-not-leak",
        api_key="provider-key-also-secret",
        auth_token="auth-token-also-secret",
    )
    text = repr(cfg)
    assert "secret-do-not-leak" not in text
    assert "provider-key-also-secret" not in text
    assert "auth-token-also-secret" not in text


def test_brave_api_key_not_in_mcp_server_config_repr() -> None:
    """env 字段值（含 BRAVE_API_KEY）必须不出现在 MCPServerConfig.__repr__ 里。"""
    cfg = brave_search_config(api_key="secret-mcp-do-not-leak")
    text = repr(cfg)
    assert "secret-mcp-do-not-leak" not in text


def test_from_env_to_build_mcp_manager_fail_fast_on_missing_key(
    monkeypatch,
) -> None:
    """端到端 fail-fast：env ENABLE_BRAVE_SEARCH=true 但无 BRAVE_API_KEY，
    走 from_env → _build_mcp_manager 应抛 RuntimeError。"""
    monkeypatch.setenv("ENABLE_BRAVE_SEARCH", "true")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    cfg = ServerConfig.from_env()
    with pytest.raises(RuntimeError, match="BRAVE_API_KEY is empty"):
        _build_mcp_manager(cfg)


# ---------------------------------------------------------------------------
# _build_mcp_manager integration
# ---------------------------------------------------------------------------


def test_build_mcp_manager_no_sources_returns_none() -> None:
    cfg = ServerConfig()
    assert _build_mcp_manager(cfg) is None


def test_build_mcp_manager_registers_brave_when_enabled() -> None:
    cfg = ServerConfig(enable_brave_search=True, brave_api_key="k")
    manager = _build_mcp_manager(cfg)

    assert manager is not None
    assert manager.get("brave-search") is not None
    # 验证 client 拿到的 cache_ttl 与默认一致
    client = manager.get("brave-search")
    assert client is not None
    assert client.cache_ttl == 60.0


def test_build_mcp_manager_brave_enabled_without_key_fails_fast() -> None:
    cfg = ServerConfig(enable_brave_search=True, brave_api_key="")
    with pytest.raises(RuntimeError, match="BRAVE_API_KEY is empty"):
        _build_mcp_manager(cfg)


def test_build_mcp_manager_combines_config_file_and_brave(tmp_path: Path) -> None:
    """mcp_config_path + enable_brave_search 同时启用 → 合到同一 MCPManager。"""
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["server.py"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = ServerConfig(
        mcp_config_path=str(config_path),
        enable_brave_search=True,
        brave_api_key="k",
    )
    manager = _build_mcp_manager(cfg)

    assert manager is not None
    assert manager.get("fs") is not None
    assert manager.get("brave-search") is not None


def test_mcp_manager_register_duplicate_name_fails_fast() -> None:
    """MCPManager.register 直接调用同名两次也应 fail-fast，不静默覆盖。"""
    from topsport_agent.mcp import MCPClient, MCPManager
    from topsport_agent.mcp.types import MCPServerConfig, MCPTransport

    cfg = MCPServerConfig(name="x", transport=MCPTransport.STDIO, command="x")
    manager = MCPManager()
    manager.register(MCPClient.from_config(cfg))
    with pytest.raises(ValueError, match="already registered"):
        manager.register(MCPClient.from_config(cfg))

    # replace=True 显式允许（caller 自负旧 client 资源关闭）
    manager.register(MCPClient.from_config(cfg), replace=True)


def test_build_mcp_manager_name_collision_with_brave_fails_fast(
    tmp_path: Path,
) -> None:
    """mcp_config_path 中已存在 brave-search → 不能再启用内置 Brave。"""
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "brave-search": {
                        "transport": "stdio",
                        "command": "/some/other/binary",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = ServerConfig(
        mcp_config_path=str(config_path),
        enable_brave_search=True,
        brave_api_key="k",
    )
    with pytest.raises(RuntimeError, match="already registered"):
        _build_mcp_manager(cfg)
