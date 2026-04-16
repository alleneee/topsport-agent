from __future__ import annotations

from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .manager import MCPManager


def build_mcp_meta_tools(manager: MCPManager) -> list[ToolSpec]:
    async def list_mcp_prompts(
        args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        server = args.get("server")
        if server:
            client = manager.get(server)
            if client is None:
                return {"error": f"server '{server}' not found"}
            try:
                prompts = await client.list_prompts()
            except Exception as exc:
                return {"server": server, "error": f"{type(exc).__name__}: {exc}"}
            return {
                "server": server,
                "prompts": [_prompt_info(p) for p in prompts],
            }

        servers: list[dict[str, Any]] = []
        for client in manager.clients():
            try:
                prompts = await client.list_prompts()
            except Exception as exc:
                servers.append(
                    {"server": client.name, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            servers.append(
                {
                    "server": client.name,
                    "prompts": [_prompt_info(p) for p in prompts],
                }
            )
        return {"servers": servers}

    async def get_mcp_prompt(
        args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        server = args["server"]
        name = args["name"]
        prompt_args = args.get("arguments") or {}
        client = manager.get(server)
        if client is None:
            return {"error": f"server '{server}' not found"}
        try:
            result = await client.get_prompt(name, prompt_args)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        messages: list[dict[str, Any]] = []
        for message in getattr(result, "messages", []) or []:
            role = getattr(message, "role", "user")
            content = getattr(message, "content", None)
            text = getattr(content, "text", None) if content is not None else None
            messages.append({"role": role, "text": text})
        return {"server": server, "name": name, "messages": messages}

    async def list_mcp_resources(
        args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        server = args.get("server")
        if server:
            client = manager.get(server)
            if client is None:
                return {"error": f"server '{server}' not found"}
            try:
                resources = await client.list_resources()
            except Exception as exc:
                return {"server": server, "error": f"{type(exc).__name__}: {exc}"}
            return {
                "server": server,
                "resources": [_resource_info(r) for r in resources],
            }

        servers: list[dict[str, Any]] = []
        for client in manager.clients():
            try:
                resources = await client.list_resources()
            except Exception as exc:
                servers.append(
                    {"server": client.name, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            servers.append(
                {
                    "server": client.name,
                    "resources": [_resource_info(r) for r in resources],
                }
            )
        return {"servers": servers}

    async def read_mcp_resource(
        args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        server = args["server"]
        uri = args["uri"]
        client = manager.get(server)
        if client is None:
            return {"error": f"server '{server}' not found"}
        try:
            result = await client.read_resource(uri)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

        parts: list[str] = []
        for content in getattr(result, "contents", []) or []:
            text = getattr(content, "text", None)
            if text is not None:
                parts.append(text)
        return {"server": server, "uri": uri, "text": "\n".join(parts)}

    return [
        ToolSpec(
            name="list_mcp_prompts",
            description=(
                "List prompts available on MCP servers. Pass 'server' to scope to one;"
                " omit to list across all configured servers."
            ),
            parameters={
                "type": "object",
                "properties": {"server": {"type": "string"}},
            },
            handler=list_mcp_prompts,
        ),
        ToolSpec(
            name="get_mcp_prompt",
            description="Render a prompt from an MCP server with the given arguments.",
            parameters={
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["server", "name"],
            },
            handler=get_mcp_prompt,
        ),
        ToolSpec(
            name="list_mcp_resources",
            description=(
                "List resources available on MCP servers. Pass 'server' to scope"
                " to one; omit to list across all configured servers."
            ),
            parameters={
                "type": "object",
                "properties": {"server": {"type": "string"}},
            },
            handler=list_mcp_resources,
        ),
        ToolSpec(
            name="read_mcp_resource",
            description="Read the content of a resource by URI from an MCP server.",
            parameters={
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "uri": {"type": "string"},
                },
                "required": ["server", "uri"],
            },
            handler=read_mcp_resource,
        ),
    ]


def _prompt_info(prompt: Any) -> dict[str, Any]:
    return {
        "name": getattr(prompt, "name", ""),
        "description": getattr(prompt, "description", "") or "",
    }


def _resource_info(resource: Any) -> dict[str, Any]:
    return {
        "uri": str(getattr(resource, "uri", "")),
        "name": getattr(resource, "name", "") or "",
        "description": getattr(resource, "description", "") or "",
        "mimeType": getattr(resource, "mimeType", None),
    }
