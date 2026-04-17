"""Prometheus metrics EventSubscriber。

可选依赖：prometheus_client。未安装则构造时抛 ImportError。
通过 `PrometheusMetrics()` 即插即用挂到 Engine.event_subscribers；`render()`
返回 text 格式 payload，server 层把它挂到 /metrics endpoint。

指标集覆盖 SLO 场景常用：
- runs_total{state}          完成 / 失败 / 取消计数
- steps_total                引擎迭代步数
- tool_calls_total{name,is_error}
- llm_tokens_total{direction=prompt|completion}
- llm_call_duration_seconds  histogram（从 LLM_CALL_START → LLM_CALL_END）

不开指标名卡底线基数（session_id / user 不入 label），避免 Prometheus 侧内存爆炸。
"""

from __future__ import annotations

import importlib
import time
from typing import Any

from ..types.events import Event, EventType


class PrometheusMetrics:
    """EventSubscriber，聚合 Engine 事件到 Prometheus metrics。

    注入自定义 registry 便于测试隔离与多实例并存；不传则用 default registry。
    """

    name = "prometheus_metrics"

    def __init__(self, *, registry: Any | None = None) -> None:
        prom_name = "prometheus_client"
        try:
            prom = importlib.import_module(prom_name)
        except ImportError as exc:
            raise ImportError(
                "prometheus_client is not installed. "
                "Run: uv sync --group metrics"
            ) from exc

        self._prom = prom
        self._registry = registry or prom.CollectorRegistry()

        self._runs_total = prom.Counter(
            "topsport_agent_runs_total",
            "Number of completed engine runs by final state.",
            labelnames=("state",),
            registry=self._registry,
        )
        self._steps_total = prom.Counter(
            "topsport_agent_steps_total",
            "Number of engine steps executed.",
            registry=self._registry,
        )
        self._tool_calls_total = prom.Counter(
            "topsport_agent_tool_calls_total",
            "Number of tool calls by name and error flag.",
            labelnames=("name", "is_error"),
            registry=self._registry,
        )
        self._llm_tokens_total = prom.Counter(
            "topsport_agent_llm_tokens_total",
            "LLM token usage by direction.",
            labelnames=("direction",),
            registry=self._registry,
        )
        self._llm_duration = prom.Histogram(
            "topsport_agent_llm_call_duration_seconds",
            "Duration of each LLM call (from LLM_CALL_START to LLM_CALL_END).",
            registry=self._registry,
        )

        # 进行中的 LLM 调用起始时间，按 session_id 隔离
        self._llm_start_by_session: dict[str, float] = {}

    async def on_event(self, event: Event) -> None:
        t = event.type
        if t == EventType.STEP_START:
            self._steps_total.inc()
        elif t == EventType.LLM_CALL_START:
            self._llm_start_by_session[event.session_id] = time.perf_counter()
        elif t == EventType.LLM_CALL_END:
            start = self._llm_start_by_session.pop(event.session_id, None)
            if start is not None:
                self._llm_duration.observe(time.perf_counter() - start)
            usage = event.payload.get("usage") or {}
            self._record_tokens(usage)
        elif t == EventType.TOOL_CALL_END:
            name = str(event.payload.get("name", "?"))
            is_error = "1" if event.payload.get("is_error") else "0"
            self._tool_calls_total.labels(name=name, is_error=is_error).inc()
        elif t == EventType.RUN_END:
            state = str(event.payload.get("final_state", "unknown"))
            self._runs_total.labels(state=state).inc()

    def _record_tokens(self, usage: dict[str, Any]) -> None:
        # 兼容 Anthropic (input_tokens / output_tokens) 与 OpenAI 字段名
        prompt = int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        completion = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        if prompt:
            self._llm_tokens_total.labels(direction="prompt").inc(prompt)
        if completion:
            self._llm_tokens_total.labels(direction="completion").inc(completion)

    def render(self) -> tuple[bytes, str]:
        """生成 /metrics endpoint 应返回的 (payload, content_type)。"""
        payload = self._prom.generate_latest(self._registry)
        content_type = self._prom.CONTENT_TYPE_LATEST
        return payload, content_type
