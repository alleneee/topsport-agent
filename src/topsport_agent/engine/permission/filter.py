"""Static tool visibility filter.

Applied in Engine._snapshot_tools. Returns only tools whose required_permissions
is a subset of the session's granted_permissions. Uses frozenset.issubset
which is O(min(len(req), len(granted))) per tool — microseconds per step.

Kill-switch and audit hooks are optional dependencies injected at construction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...types.session import Session
    from ...types.tool import ToolSpec
    from .audit import AuditLogger
    from .killswitch import KillSwitchGate

_logger = logging.getLogger(__name__)


class ToolVisibilityFilter:
    """Static capability filter for tool pools.

    - Kill-switch first: if the tenant is killed, returns [].
    - Set subset check: tool.required_permissions.issubset(session.granted_permissions).
    - Audit logging of filtered-out tools is optional and non-blocking.
    """

    def __init__(
        self,
        *,
        audit_logger: "AuditLogger | None" = None,
        kill_switch: "KillSwitchGate | None" = None,
    ) -> None:
        self._audit = audit_logger
        self._kill = kill_switch

    def filter(
        self,
        pool: "list[ToolSpec]",
        session: "Session",
    ) -> "list[ToolSpec]":
        if self._kill is not None and self._kill.is_active(session.tenant_id):
            _logger.info(
                "kill-switch active for tenant %r; returning empty tool pool",
                session.tenant_id,
            )
            if self._audit is not None:
                self._audit.log_killswitch_blocked(session, pool)
            return []

        granted = session.granted_permissions
        visible: list[ToolSpec] = []
        filtered: list[ToolSpec] = []
        for tool in pool:
            if tool.required_permissions.issubset(granted):
                visible.append(tool)
            else:
                filtered.append(tool)

        if filtered and self._audit is not None:
            self._audit.log_filtered(session, filtered)

        return visible
