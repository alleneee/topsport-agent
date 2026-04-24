def test_gateway_protocols_importable() -> None:
    from topsport_agent.database.gateway import Database, Transaction  # noqa: F401


def test_gateway_has_required_attrs() -> None:
    from topsport_agent.database.gateway import Database, Transaction

    for proto in (Database, Transaction):
        names = set(dir(proto))
        assert "execute" in names, f"{proto.__name__} missing execute"
        assert "fetch_one" in names, f"{proto.__name__} missing fetch_one"
        assert "fetch_all" in names, f"{proto.__name__} missing fetch_all"
        assert "fetch_val" in names, f"{proto.__name__} missing fetch_val"

    # Only Database has lifecycle + dialect + transaction factory
    db_names = set(dir(Database))
    for required in ("dialect", "connect", "close", "health_check", "transaction"):
        assert required in db_names, f"Database missing {required}"
