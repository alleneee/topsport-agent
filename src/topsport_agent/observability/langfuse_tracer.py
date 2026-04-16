from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from ..types.events import Event, EventType

_logger = logging.getLogger(__name__)


class LangfuseTracer:
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
    ) -> None:
        self._flush_on_run_end = flush_on_run_end
        self._state_by_session: dict[str, dict[str, Any]] = {}

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
            kwargs["base_url"] = resolved_base
        if resolved_environment:
            kwargs["environment"] = resolved_environment
        if resolved_release:
            kwargs["release"] = resolved_release

        self._client = Langfuse(**kwargs)

    async def on_event(self, event: Event) -> None:
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
            pass

    def _state(self, session_id: str) -> dict[str, Any]:
        return self._state_by_session.setdefault(session_id, {"tools": {}})

    def _handle_run_start(self, event: Event) -> None:
        state = self._state(event.session_id)
        root = self._client.start_observation(
            name=f"agent.run[{event.session_id}]",
            as_type="agent",
            input=event.payload,
        )
        try:
            root.update_trace(
                session_id=event.session_id,
                input=event.payload,
            )
        except Exception:
            pass
        state["root"] = root

    def _handle_run_end(self, event: Event) -> None:
        state = self._state_by_session.get(event.session_id)
        if state is None:
            return
        root = state.get("root")
        if root is not None:
            try:
                root.update(output=event.payload)
            except Exception:
                pass
            try:
                root.update_trace(output=event.payload)
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
            input=event.payload,
        )
        state["step"] = span

    def _handle_step_end(self, event: Event) -> None:
        state = self._state(event.session_id)
        span = state.pop("step", None)
        if span is not None:
            try:
                span.update(output=event.payload)
            except Exception:
                pass
            try:
                span.end()
            except Exception:
                pass

    def _handle_llm_start(self, event: Event) -> None:
        state = self._state(event.session_id)
        parent = state.get("step") or state.get("root") or self._client
        gen = parent.start_observation(
            name="llm.call",
            as_type="generation",
            model=event.payload.get("model"),
            input=event.payload,
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
            pass
        try:
            gen.end()
        except Exception:
            pass

    def _handle_tool_start(self, event: Event) -> None:
        state = self._state(event.session_id)
        parent = state.get("step") or state.get("root") or self._client
        tool_span = parent.start_observation(
            name=f"tool.{event.payload.get('name', 'unknown')}",
            as_type="tool",
            input=event.payload,
        )
        state["tools"][event.payload["call_id"]] = tool_span

    def _handle_tool_end(self, event: Event) -> None:
        state = self._state(event.session_id)
        tool_span = state["tools"].pop(event.payload["call_id"], None)
        if tool_span is None:
            return
        update_kwargs: dict[str, Any] = {"output": event.payload}
        if event.payload.get("is_error"):
            update_kwargs["level"] = "ERROR"
            update_kwargs["status_message"] = f"tool {event.payload.get('name')} errored"
        try:
            tool_span.update(**update_kwargs)
        except Exception:
            pass
        try:
            tool_span.end()
        except Exception:
            pass

    def _handle_error(self, event: Event) -> None:
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
                pass

    def _handle_cancelled(self, event: Event) -> None:
        state = self._state_by_session.get(event.session_id)
        if state is None:
            return
        root = state.get("root")
        if root is not None:
            try:
                root.update(level="WARNING", status_message="cancelled")
            except Exception:
                pass

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
