from __future__ import annotations

from typing import Any

from ..types.tool import ToolContext, ToolSpec
from .matcher import SkillMatcher
from .registry import SkillRegistry


def build_skill_tools(
    registry: SkillRegistry,
    matcher: SkillMatcher,
) -> list[ToolSpec]:
    async def load_skill(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        name = args["name"]
        manifest = registry.get(name)
        if manifest is None:
            return {
                "ok": False,
                "error": f"skill '{name}' not found",
                "available": [m.name for m in registry.list()],
            }
        matcher.activate(ctx.session_id, name)
        return {
            "ok": True,
            "name": name,
            "description": manifest.description,
            "resource_count": len(manifest.resources),
        }

    async def unload_skill(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        name = args["name"]
        deactivated = matcher.deactivate(ctx.session_id, name)
        return {"ok": deactivated, "name": name}

    async def list_skills(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        manifests = registry.list()
        active = set(matcher.active_skills(ctx.session_id))
        return {
            "count": len(manifests),
            "skills": [
                {
                    "name": manifest.name,
                    "description": manifest.description,
                    "active": manifest.name in active,
                }
                for manifest in manifests
            ],
        }

    load_spec = ToolSpec(
        name="load_skill",
        description=(
            "Activate a skill's full instructions for the current session. "
            "Inspect available skills with list_skills first."
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        handler=load_skill,
    )

    unload_spec = ToolSpec(
        name="unload_skill",
        description="Deactivate a previously loaded skill for the current session.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        handler=unload_skill,
    )

    list_spec = ToolSpec(
        name="list_skills",
        description="List every registered skill with name, description, and active state.",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=list_skills,
    )

    return [load_spec, unload_spec, list_spec]
