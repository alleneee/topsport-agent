"""Engine permission sub-package.

v2 (capability-based): ToolVisibilityFilter, KillSwitchGate, AuditLogger,
PersonaRegistry, etc. See docs/superpowers/specs/2026-04-23-permission-
capability-acl-design.md.

v1 (deprecated runtime decision) symbols are re-exported from `_legacy`
for one minor release; they emit DeprecationWarning via the types module.
"""

from __future__ import annotations

# Legacy v1 re-exports (deprecated)
from ._legacy import (
    AlwaysAskAsker,
    AlwaysDenyAsker,
    DefaultPermissionChecker,
)

__all__ = [
    "AlwaysAskAsker",
    "AlwaysDenyAsker",
    "DefaultPermissionChecker",
]
