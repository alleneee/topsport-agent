def test_ratelimiter_protocol_importable() -> None:
    from topsport_agent.ratelimit.limiter import RateLimiter  # noqa: F401


def test_ratelimiter_has_check_method() -> None:
    from topsport_agent.ratelimit.limiter import RateLimiter

    assert "check" in dir(RateLimiter)
