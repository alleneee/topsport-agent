"""Structured sub-dataclasses for AgentConfig.

`AgentConfig` historically grew into 22+ flat fields covering four orthogonal
concerns: agent identity, capability toggles, registries (extras + permission
+ multimodal), and engine-level options. Tests had to fill 8-keyword mocks;
new readers had to scan the whole class body to know what was a "name field"
vs an "audit hook".

Phase 4 introduces three small dataclasses representing the natural concerns:

  AgentIdentity      — name / description / system_prompt / model / max_steps
  CapabilityToggles  — enable_* flags + filesystem path config for those flags
  CapabilityRegistry — extras (tools, hooks, modules) + permission objects +
                       sanitizer / image_generator / provider_options

These types are public API. Operators can construct them independently and
hand them to `AgentConfig.from_parts(identity=..., toggles=..., registry=...)`,
which produces a flat `AgentConfig` with all underlying fields populated.

Backward compatibility: `AgentConfig` keeps its original flat keyword
constructor and all 22 fields. Existing call sites (`AgentConfig(name="x",
model="y", ...)`) and field accesses (`config.name`, `config.enable_skills`)
continue to work unchanged. The flat fields remain the storage source of
truth in this PR; sub-dataclasses are exposed as derived views via
`AgentConfig.identity` / `.toggles` / `.registry` properties.

This split follows the same pattern as splitting a CLAUDE.md into "agent
identity" + "settings (hooks/permissions)" — concerns separated, but the
on-disk format stays compatible.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..engine.hooks import (
        ContextProvider,
        EventSubscriber,
        PostStepHook,
        PostToolUseHook,
        PreToolUseHook,
        ToolSource,
    )
    from ..engine.permission.audit import AuditLogger
    from ..engine.permission.filter import ToolVisibilityFilter
    from ..engine.permission.persona_registry import PersonaRegistry
    from ..engine.sanitizer import ToolResultSanitizer
    from ..llm.image_generation import OpenAIImageGenerationClient
    from ..types.permission import PermissionAsker, PermissionChecker, Persona
    from ..types.tool import ToolSpec
    from .capabilities import CapabilityModule


# Default max_steps shared by AgentIdentity and AgentConfig — single source of
# truth so the schema-guard test (test_part_field_names_partition_agent_config_fields_exactly)
# never has to chase a drift caused by changing one and forgetting the other.
DEFAULT_MAX_STEPS: int = 20


@dataclass(slots=True, frozen=True)
class AgentIdentity:
    """Who the Agent is. Stable across capability/registry permutations.

    Used to differentiate one preset (default / browser / planner) from
    another. Empty defaults are provided so partial mocks are ergonomic;
    in production these should all be set explicitly.
    """

    name: str = ""
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    max_steps: int = DEFAULT_MAX_STEPS


@dataclass(slots=True, frozen=True)
class CapabilityToggles:
    """Which capabilities mount on this Agent.

    Each `enable_*` is a fail-closed gate: leaving it default means the
    matching CapabilityModule does not run, regardless of what's in the
    operator's environment (the principle behind every breach prevention
    review since this codebase began). Path fields configure the filesystem
    layout used by the corresponding modules when their gate is on.
    """

    enable_skills: bool = True
    enable_memory: bool = True
    enable_plugins: bool = True
    enable_browser: bool = False
    enable_file_ops: bool = False
    stream: bool = False
    memory_base_path: Path | None = None
    local_skill_dirs: list[Path] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class CapabilityRegistry:
    """Everything that gets handed to / wired into the Agent at construction:
    additional tools/hooks, capability modules, permission objects, and the
    optional engine-level singletons (sanitizer / image_generator /
    provider_options).

    All fields default to empty / None so mocks can construct an empty
    registry and only fill what's relevant. Operators in production
    typically build this once at startup (per tenant or per persona).
    """

    # Extras — appended to the corresponding CapabilityBundle list during
    # Agent.from_config; merged in registration order before topo-sorted
    # modules contribute their own bundles.
    extra_tools: list["ToolSpec"] = field(default_factory=list)
    extra_context_providers: list["ContextProvider"] = field(default_factory=list)
    extra_tool_sources: list["ToolSource"] = field(default_factory=list)
    extra_post_step_hooks: list["PostStepHook"] = field(default_factory=list)
    extra_event_subscribers: list["EventSubscriber"] = field(default_factory=list)
    extra_pre_tool_hooks: list["PreToolUseHook"] = field(default_factory=list)
    extra_post_tool_hooks: list["PostToolUseHook"] = field(default_factory=list)
    extra_capability_modules: list["CapabilityModule"] = field(default_factory=list)

    # Permission wiring (v2 capability-ACL).
    persona: "Persona | str | None" = None
    persona_registry: "PersonaRegistry | None" = None
    tenant_id: str | None = None
    permission_filter: "ToolVisibilityFilter | None" = None
    audit_logger: "AuditLogger | None" = None
    permission_checker: "PermissionChecker | None" = None
    permission_asker: "PermissionAsker | None" = None

    # Engine-level singletons not modeled as CapabilityModule (yet).
    sanitizer: "ToolResultSanitizer | None" = None
    image_generator: "OpenAIImageGenerationClient | None" = None
    provider_options: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers used by AgentConfig to view-from / construct-via parts
# ---------------------------------------------------------------------------


def identity_field_names() -> tuple[str, ...]:
    """Field names owned by AgentIdentity, derived from the dataclass itself
    so the view/round-trip code never drifts away from the schema."""
    return tuple(f.name for f in dataclasses.fields(AgentIdentity))


def toggle_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(CapabilityToggles))


def registry_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(CapabilityRegistry))


def isolate_value(value: Any) -> Any:
    """Shallow-copy mutable container values so view→config and
    config→view passes don't share aliased lists/dicts.

    Why: `AgentConfig.identity / .toggles / .registry` should be
    snapshots — mutating a view should not silently rewrite the parent
    config; symmetrically, mutating a CapabilityRegistry passed into
    `from_parts` should not later show up in the constructed AgentConfig.
    Lists and dicts get a fresh container; nested objects are still
    shared by reference (cheap, and matches Python's general semantic of
    "don't deep-copy unless asked"). Frozen dataclasses already prevent
    the scalar leak path, this closes the container path.

    Public API: imported by `agent/base.py` to wire the views and
    `from_parts`; exported via `__all__` so downstream code building
    its own AgentConfig views can reuse the same shallow-copy contract
    rather than reinventing snapshot semantics.
    """
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


__all__ = [
    "DEFAULT_MAX_STEPS",
    "AgentIdentity",
    "CapabilityRegistry",
    "CapabilityToggles",
    "identity_field_names",
    "isolate_value",
    "registry_field_names",
    "toggle_field_names",
]
