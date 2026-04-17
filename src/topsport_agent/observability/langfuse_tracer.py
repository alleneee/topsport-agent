from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Iterable
from typing import Any

from ..types.events import Event, EventType
from .redaction import Redactor, validate_base_url

_logger = logging.getLogger(__name__)


def _identity(v: Any) -> Any:
    return v


class LangfuseTracer:
    """事件驱动的 Langfuse 追踪器：span 生命周期跨越多个异步事件，不能用 with 块。

    使用 start_observation + end 的显式 API（v3），兼容 v4 的统一接口。

    H-S2 脱敏：redactor 在 payload 进入 start_observation/update/update_trace
    之前运行；默认恒等函数（不改变历史行为），生产建议注入 SimpleRedactor。
    allowed_base_urls 非空时强制 base_url 前缀白名单，避免追踪流量被导向任意目标。
    """
    name = "langfuse"

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        environment: str | None = None,
        release: str | None = None,
        client: Any | None = None,
        flush_on_run_end: bool = True,
        redactor: Redactor | None = None,
        allowed_base_urls: Iterable[str] = (),
    ) -> None:
        self._flush_on_run_end = flush_on_run_end
        self._redact: Callable[[Any], Any] = redactor or _identity
        # 按 session_id 隔离追踪状态，支持并发多 session。
        self._state_by_session: dict[str, dict[str, Any]] = {}

        # 测试路径：直接注入 mock client；生产路径：延迟 import langfuse。
        if client is not None:
            self._client = client
            return

        module_name = "langfuse"
        try:
            langfuse_module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                "langfuse is not installed. Run: uv sync --group tracing"
            ) from exc
        Langfuse = langfuse_module.Langfuse

        kwargs: dict[str, Any] = {}
        resolved_public = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        resolved_secret = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        resolved_base = base_url or os.getenv("LANGFUSE_BASE_URL")
        resolved_environment = environment or os.getenv("LANGFUSE_ENVIRONMENT")
        resolved_release = release or os.getenv("LANGFUSE_RELEASE")

        if resolved_public:
            kwargs["public_key"] = resolved_public
        if resolved_secret:
            kwargs["secret_key"] = resolved_secret
        if resolved_base:
            # base_url 白名单校验在真正构造 Langfuse 客户端之前，避免先泄露一次。
            validate_base_url(resolved_base, allowed_base_urls)
            kwargs["base_url"] = resolved_base
        if resolved_environment:
            kwargs["environment"] = resolved_environment
        if resolved_release:
            kwargs["release"] = resolved_release

        self._client = Langfuse(**kwargs)

    async def on_event(self, event: Event) -> None:
        """事件派发表模式：O(1) 查找，新事件类型只需在 _handlers 表末尾加一行。"""
        try:
            handler = self._handlers.get(event.type)
            if handler is not None:
                handler(self, event)
        except Exception as exc:
            _logger.warning(
                "langfuse tracer failed on %s: %r", event.type.value, exc
            )

    def shutdown(self) -> None:
        try:
            self._client.flush()
        except Exception:
            _logger.debug("langfuse operation failed", exc_info=True)

    def _state(self, session_id: str) -> dict[str, Any]:
        return self._state_by_session.setdefault(session_id, {"tools": {}})

    def _handle_run_start(self, event: Event) -> None:
        state = self._state(event.session_id)
        root = self._client.start_observation(
            name=f"agent.run[{event.session_id}]",
            as_type="agent",
            input=self._redact(event.payload),
        )
        try:
            root.update_trace(
                session_id=event.session_id,
                input=self._redact(event.payload),
            )
        except Exception:
            _logger.debug("langfuse operation failed", exc_info=True)
        state["root"] = root

    def _handle_run_end(self, event: Event) -> None:
        """RUN_END 关闭根 span 并清理 session 状态，flush 保证 trace 落盘。"""
        state = self._state_by_session.get(event.session_id)
        if state is None:
            return
        root = state.get("root")
        if root is not None:
            try:
                root.update(output=self._redact(event.payload))
            except Exception:
                pass
            try:
                root.update_trace(output=self._redact(event.payload))
            except Exception:
                pass
            try:
                root.end()
            except Exception:
                pass
        self._state_by_session.pop(event.session_id, None)
        if self._flush_on_run_end:
            try:
                self._client.flush()
            except Exception:
                pass

    def _handle_step_start(self, event: Event) -> None:
        state = self._state(event.session_id)
        parent = state.get("root") or self._client
        span = parent.start_observation(
            name=f"step.{event.payload.get('step', 0)}",
            as_type="span",
            input=self._redact(event.payload),
        )
        state["step"] = span

    def _handle_step_end(self, event: Event) -> None:
        state = self._state(event.session_id)
        span = state.pop("step", None)
        if span is not None:
            try:
                span.update(output=self._redact(event.payload))
            except Exception:
                pass
            try:
                span.end()
            except Exception:
                pass

    def _handle_llm_start(self, event: Event) -> None:
        """LLM 调用挂在当前 step 下；若无 step 则挂在 root 下，保证 trace 树结构正确。"""
        state = self._state(event.session_id)
        parent = state.get("step") or state.get("root") or self._client
        gen = parent.start_observation(
            name="llm.call",
            as_type="generation",
            model=event.payload.get("model"),
            input=self._redact(event.payload),
        )
        state["llm"] = gen

    def _handle_llm_end(self, event: Event) -> None:
        state = self._state(event.session_id)
        gen = state.pop("llm", None)
        if gen is None:
            return
        usage = event.payload.get("usage") or {}
        update_kwargs: dict[str, Any] = {
            "output": {
                "tool_call_count": event.payload.get("tool_call_count"),
                "finish_reason": event.payload.get("finish_reason"),
            }
        }
        if usage:
            update_kwargs["usage_details"] = usage
        try:
            gen.update(**update_kwargs)
        except Exception:
            _logger.debug("langfuse operation failed", exc_info=True)
        try:
            gen.end()
        except Exception:
            _logger.debug("langfuse operation failed", exc_info=True)

    def _handle_tool_start(self, event: Event) -> None:
        state = self._state(event.session_id)
        parent = state.get("step") or state.get("root") or self._client
        tool_span = parent.start_observation(
            name=f"tool.{event.payload.get('name', 'unknown')}",
            as_type="tool",
            input=self._redact(event.payload),
        )
        state["tools"][event.payload["call_id"]] = tool_span

    def _handle_tool_end(self, event: Event) -> None:
        state = self._state(event.session_id)
        tool_span = state["tools"].pop(event.payload["call_id"], None)
        if tool_span is None:
            return
        update_kwargs: dict[str, Any] = {"output": self._redact(event.payload)}
        if event.payload.get("is_error"):
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = f"tool {event.payload.get('name')} errored"
        try:
            tool_span.update(**update_kwargs)
        except Exception:
            _logger.debug("langfuse operation failed", exc_info=True)
        try:
            tool_span.end()
        except Exception:
            _logger.debug("langfuse operation failed", exc_info=True)

    def _handle_error(self, event: Event) -> None:
        """异常和取消都要清理 session 状态，避免内存泄漏。"""
        state = self._state_by_session.get(event.session_id)
        if state is None:
            return
        root = state.get("root")
        if root is not None:
            try:
                root.update(
                    level="ERROR",
                    status_message=event.payload.get("message", ""),
                )
            except Exception:
                _logger.debug("langfuse error update failed", exc_info=True)
        self._state_by_session.pop(event.session_id, None)

    def _handle_cancelled(self, event: Event) -> None:
        state = self._state_by_session.get(event.session_id)
        if state is None:
            return
        root = state.get("root")
        if root is not None:
            try:
                root.update(level="WARNING", status_message="cancelled")
            except Exception:
                _logger.debug("langfuse cancel update failed", exc_info=True)
        self._state_by_session.pop(event.session_id, None)

    # 事件类型 -> 处理方法的静态派发表；顺序无关，dict 查找 O(1)。
    _handlers: dict[EventType, Any] = {
        EventType.RUN_START: _handle_run_start,
        EventType.RUN_END: _handle_run_end,
        EventType.STEP_START: _handle_step_start,
        EventType.STEP_END: _handle_step_end,
        EventType.LLM_CALL_START: _handle_llm_start,
        EventType.LLM_CALL_END: _handle_llm_end,
        EventType.TOOL_CALL_START: _handle_tool_start,
        EventType.TOOL_CALL_END: _handle_tool_end,
        EventType.ERROR: _handle_error,
        EventType.CANCELLED: _handle_cancelled,
    }
