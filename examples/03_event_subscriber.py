"""示例 03：事件订阅 —— 观测 / 日志 / 自定义 tracing。

EventSubscriber 约定：
- 拥有 name 属性
- 实现 async on_event(event)
- 单个 subscriber 抛异常不会影响 Engine 和其他 subscriber
- 事件按 yield 顺序串行回调（见 loop.py:_emit）

运行：
    uv sync --group llm
    ANTHROPIC_API_KEY=sk-... uv run python examples/03_event_subscriber.py
"""

from __future__ import annotations

import asyncio
import time

from topsport_agent.agent import default_agent
from topsport_agent.llm.providers.anthropic import AnthropicProvider
from topsport_agent.types.events import Event, EventType


class LatencyTracer:
    """记录每一步 LLM / Tool 调用耗时，并在 run 结束时打印统计。"""

    name = "latency-tracer"

    def __init__(self) -> None:
        self._llm_start: float | None = None
        self._tool_start: dict[str, float] = {}
        self._stats: list[tuple[str, float]] = []

    async def on_event(self, event: Event) -> None:
        now = time.perf_counter()
        if event.type == EventType.LLM_CALL_START:
            self._llm_start = now
        elif event.type == EventType.LLM_CALL_END and self._llm_start is not None:
            self._stats.append(("llm", now - self._llm_start))
            self._llm_start = None
        elif event.type == EventType.TOOL_CALL_START:
            self._tool_start[event.payload["call_id"]] = now
        elif event.type == EventType.TOOL_CALL_END:
            start = self._tool_start.pop(event.payload["call_id"], None)
            if start is not None:
                self._stats.append((f"tool:{event.payload['name']}", now - start))
        elif event.type == EventType.RUN_END:
            print("\n=== Latency summary ===")
            for name, dt in self._stats:
                print(f"  {name:<30s} {dt * 1000:.1f} ms")


async def main() -> None:
    provider = AnthropicProvider()
    agent = default_agent(provider, "claude-sonnet-4-6", enable_browser=False)
    # 把 tracer 追加到 engine（AgentConfig.extra_event_subscribers 也可以）
    agent.engine._event_subscribers.append(LatencyTracer())

    try:
        session = agent.new_session()
        async for _ in agent.run("用一句话自我介绍。", session):
            pass
        print("ASSISTANT:", session.messages[-1].content)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
