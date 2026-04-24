import importlib.util

import pytest

from topsport_agent.ratelimit.metrics import RateLimitMetrics

_prom_available = importlib.util.find_spec("prometheus_client") is not None


def test_metrics_can_be_constructed_without_prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    import importlib
    import topsport_agent.ratelimit.metrics as mod
    importlib.reload(mod)

    metrics = mod.RateLimitMetrics()
    # Should not raise even without prometheus_client
    metrics.inc_request("ip")
    metrics.inc_denied("tenant")
    metrics.inc_degraded("ConnectionError")
    metrics.observe_check_duration(0.001)


@pytest.mark.skipif(not _prom_available, reason="prometheus-client not installed")
def test_metrics_with_prometheus_registers_counters() -> None:
    from prometheus_client import CollectorRegistry

    metrics = RateLimitMetrics(registry=CollectorRegistry())
    metrics.inc_request("ip")
    metrics.inc_request("ip")
    metrics.inc_denied("tenant")
    # Calls must not raise.
