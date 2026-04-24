"""Verify ENABLE_DATABASE flag drives lifespan wiring."""
from __future__ import annotations

import pytest

from topsport_agent.database.backends.null import NullGateway
from topsport_agent.server.app import create_app
from topsport_agent.server.config import ServerConfig


def _stub_provider():
    class _Prov:
        name = "stub"

        async def complete(self, request):  # pragma: no cover
            raise RuntimeError("not used in lifespan tests")

    return _Prov()


@pytest.mark.asyncio
async def test_lifespan_disabled_database_uses_null_gateway() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_database=False,
    )
    app = create_app(cfg, provider=_stub_provider())
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.database, NullGateway)
        assert await app.state.database.health_check() is True


@pytest.mark.asyncio
async def test_lifespan_enabled_without_url_raises() -> None:
    cfg = ServerConfig(
        api_key="dummy",
        default_model="test",
        auth_required=False,
        enable_database=True,
        database_backend="postgres",
        database_url=None,
    )
    app = create_app(cfg, provider=_stub_provider())
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        async with app.router.lifespan_context(app):
            pass
