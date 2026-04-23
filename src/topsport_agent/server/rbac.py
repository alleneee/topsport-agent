"""RBAC dependency for FastAPI routes on the admin API.

Gates endpoints at the HTTP layer. The Engine never sees RBAC — by the
time a call reaches the engine, the caller's authority has been verified
by this middleware.

Role hierarchy (higher includes lower):
    ADMIN  >  OPERATOR  >  AUDITOR  >  AGENT

`require_role(Role.OPERATOR)` permits ADMIN and OPERATOR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status

from ..types.permission import Role

__all__ = ["RBACPrincipal", "require_role"]


_HIERARCHY: dict[Role, int] = {
    Role.AGENT: 0,
    Role.AUDITOR: 1,
    Role.OPERATOR: 2,
    Role.ADMIN: 3,
}


@dataclass(frozen=True, slots=True)
class RBACPrincipal:
    """The caller identity after auth middleware. Injected as a dependency."""
    user_id: str
    tenant_id: str
    role: Role


def _default_principal_resolver() -> RBACPrincipal:
    """Placeholder: production deployments override via dependency_overrides
    or plug in a real JWT/session decoder here."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="no RBAC principal resolver configured",
    )


def require_role(min_role: Role):
    """FastAPI dependency factory. Raises 403 if caller's role is below min."""

    def checker(
        principal: Annotated[RBACPrincipal, Depends(_default_principal_resolver)],
    ) -> RBACPrincipal:
        if _HIERARCHY[principal.role] < _HIERARCHY[min_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role {principal.role.value} insufficient; requires {min_role.value}",
            )
        return principal

    return Depends(checker)
