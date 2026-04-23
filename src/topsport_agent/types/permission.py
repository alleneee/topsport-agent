"""Capability-based permission model for enterprise internal AI platform.

Design: tools declare `required_permissions`; sessions carry `granted_permissions`
populated from a Persona. Runtime enforcement is a static frozenset subset check
executed once per `Engine._snapshot_tools` via `ToolVisibilityFilter`.

Legacy symbols (`PermissionBehavior`, `PermissionDecision`, `PermissionChecker`,
`PermissionAsker`, `allow`, `deny`, `ask`) from the previous runtime-decision
design are retained for one minor release with a DeprecationWarning on import
and will be removed in the next release.
"""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from .message import ToolCall
    from .tool import ToolContext, ToolSpec

__all__ = [
    # Core v2 model
    "AuditEntry",
    "Permission",
    "Persona",
    "PersonaAssignment",
    "Role",
    # Deprecated v1 symbols
    "PermissionAsker",
    "PermissionBehavior",
    "PermissionChecker",
    "PermissionDecision",
    "allow",
    "ask",
    "deny",
]


# ---------------------------------------------------------------------------
# Core v2: capability model
# ---------------------------------------------------------------------------


class Permission(StrEnum):
    """Predefined capability set. New permissions are added by extending this
    enum or by passing plain strings; the filter treats any string that
    satisfies the StrEnum contract as a permission identifier."""

    # Filesystem
    FS_READ = "fs.read"
    FS_WRITE = "fs.write"
    # Shell (sandbox-confined)
    SHELL_SAFE = "shell.safe"
    SHELL_FULL = "shell.full"
    # Network
    NETWORK_OUT = "network.out"
    # MCP servers — extend as platform grows
    MCP_GITHUB = "mcp.github"
    MCP_JIRA = "mcp.jira"
    MCP_ZENDESK = "mcp.zendesk"
    MCP_SNOWFLAKE = "mcp.snowflake"
    MCP_SAP = "mcp.sap"
    # Meta
    MEMORY_WRITE = "memory.write"
    AGENT_SPAWN = "agent.spawn"


class Role(StrEnum):
    """HTTP admin API access roles. Separate from session end-user identity."""

    ADMIN = "admin"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class Persona:
    id: str
    display_name: str
    description: str
    permissions: frozenset[Permission]
    version: int = 1


@dataclass(frozen=True, slots=True)
class PersonaAssignment:
    tenant_id: str
    persona_ids: frozenset[str]
    default_persona_id: str | None
    user_id: str | None = None
    group_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuditEntry:
    id: str
    tenant_id: str
    session_id: str
    user_id: str | None
    persona_id: str | None
    tool_name: str
    tool_required: frozenset[Permission]
    subject_granted: frozenset[Permission]
    outcome: Literal["allowed", "filtered_out", "error", "killed"]
    args_preview: dict[str, Any]
    reason: str | None
    timestamp: datetime
    # Reserved for future quota / cost tracking; populated as 0 / None in v1.
    cost_tokens: int = 0
    cost_latency_ms: int = 0
    group_id: str | None = None


# ---------------------------------------------------------------------------
# DEPRECATED v1 symbols — removed in next minor release
#
# Implementation note: legacy symbols are defined with a leading underscore
# and exposed via module `__getattr__` so that every `from ... import X`
# access emits a DeprecationWarning (module-level `warnings.warn` would fire
# only once per process — too early if the module is imported transitively
# before any consumer sees the warning).
# ---------------------------------------------------------------------------


class _PermissionBehavior(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(slots=True, frozen=True)
class _PermissionDecision:
    behavior: _PermissionBehavior
    reason: str | None = None
    updated_input: dict[str, Any] | None = None


def _allow(updated_input: dict[str, Any] | None = None) -> _PermissionDecision:
    return _PermissionDecision(_PermissionBehavior.ALLOW, updated_input=updated_input)


def _deny(reason: str) -> _PermissionDecision:
    return _PermissionDecision(_PermissionBehavior.DENY, reason=reason)


def _ask(reason: str | None = None) -> _PermissionDecision:
    return _PermissionDecision(_PermissionBehavior.ASK, reason=reason)


class _PermissionChecker(Protocol):
    name: str

    async def check(
        self,
        tool: "ToolSpec",
        call: "ToolCall",
        context: "ToolContext",
    ) -> _PermissionDecision: ...


class _PermissionAsker(Protocol):
    name: str

    async def ask(
        self,
        tool: "ToolSpec",
        call: "ToolCall",
        context: "ToolContext",
        reason: str | None,
    ) -> _PermissionDecision: ...


PermissionCheckFn = Callable[
    ["ToolSpec", "ToolCall", "ToolContext"], Awaitable[_PermissionDecision]
]


_LEGACY_ALIASES: dict[str, Any] = {
    "PermissionBehavior": _PermissionBehavior,
    "PermissionDecision": _PermissionDecision,
    "PermissionChecker": _PermissionChecker,
    "PermissionAsker": _PermissionAsker,
    "allow": _allow,
    "deny": _deny,
    "ask": _ask,
}


def __getattr__(name: str) -> Any:
    if name in _LEGACY_ALIASES:
        warnings.warn(
            "topsport_agent.types.permission: PermissionBehavior/Checker/Asker/"
            "Decision and allow/deny/ask helpers are deprecated; migrate to the "
            "capability-based Permission / Persona / PersonaAssignment model. "
            "See docs/superpowers/specs/2026-04-23-permission-capability-acl-"
            "design.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _LEGACY_ALIASES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
