from __future__ import annotations

import json
from pathlib import Path

import pytest

from topsport_agent.mcp.config import load_mcp_config
from topsport_agent.mcp.tool_bridge import MCPToolSource
from topsport_agent.types.permission import Permission


def test_mcp_config_parses_permissions(tmp_path: Path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text(json.dumps({
        "mcpServers": {
            "github": {
                "transport": "stdio",
                "command": "mcp-github",
                "permissions": ["mcp.github"],
            }
        }
    }))
    configs = load_mcp_config(cfg_path)
    assert len(configs) == 1
    assert "mcp.github" in configs[0].permissions


def test_mcp_config_permissions_default_empty(tmp_path: Path):
    cfg_path = tmp_path / "mcp.json"
    cfg_path.write_text(json.dumps({
        "mcpServers": {
            "x": {"transport": "stdio", "command": "mcp-x"}
        }
    }))
    configs = load_mcp_config(cfg_path)
    assert configs[0].permissions == frozenset()


@pytest.mark.asyncio
async def test_mcp_tool_bridge_propagates_permissions():
    """Bridged ToolSpec.required_permissions includes the server's permissions."""
    from topsport_agent.types.tool import ToolContext

    # Minimal fake MCPClient that returns one tool
    class _FakeClient:
        def __init__(self) -> None:
            self.name = "github"
            self.permissions = frozenset({Permission.MCP_GITHUB})
        async def list_tools(self):
            from topsport_agent.mcp.types import MCPToolInfo
            return [MCPToolInfo(
                name="search_issues",
                description="",
                parameters={"type": "object"},
            )]
        async def call_tool(self, name, args):
            return {"content": [{"type": "text", "text": "ok"}]}

    source = MCPToolSource(_FakeClient())  # type: ignore[arg-type]
    tools = await source.list_tools()
    assert len(tools) == 1
    assert Permission.MCP_GITHUB in tools[0].required_permissions
