from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..llm.provider import LLMProvider, LLMResponse, StreamingLLMProvider
from ..llm.request import LLMRequest
from ..llm.response import wrap_response_metadata
from ..types.events import Event, EventType
from ..types.message import Message, Role, ToolCall, ToolResult
from ..types.session import RunState, Session
from ..types.tool import ToolContext, ToolSpec
from .hooks import ContextProvider, EventSubscriber, PostStepHook, ToolSource
from .prompt import PromptBuilder, SectionPriority
from .sanitizer import SECURITY_GUARD_CONTENT, SECURITY_GUARD_TAG, ToolResultSanitizer

_logger = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


class BudgetExceeded(Exception):
    """Session.token_budget 被突破时抛出，Engine 转 RunState.ERROR。"""


def _accumulate_usage(session: Any, usage: dict[str, Any] | None) -> None:
    """兼容 Anthropic (input_tokens/output_tokens) 和 OpenAI (prompt_tokens/
    completion_tokens) 两种字段名；都缺就退化用 total_tokens。"""
    if not usage:
        return
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        total = int(usage.get("prompt_tokens", 0) or 0) + int(
            usage.get("completion_tokens", 0) or 0
        )
    elif "input_tokens" in usage or "output_tokens" in usage:
        total = int(usage.get("input_tokens", 0) or 0) + int(
            usage.get("output_tokens", 0) or 0
        )
    else:
        total = int(usage.get("total_tokens", 0) or 0)
    session.token_spent += total


@dataclass(slots=True)
class EngineConfig:
    model: str
    max_steps: int = 20
    provider_options: dict[str, Any] | None = None
    # 是否使用流式调用。只有在 provider 也实现 StreamingLLMProvider 时才真正生效。
    stream: bool = False


class Engine:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[ToolSpec],
        config: EngineConfig,
        *,
        context_providers: list[ContextProvider] | None = None,
        tool_sources: list[ToolSource] | None = None,
        post_step_hooks: list[PostStepHook] | None = None,
        event_subscribers: list[EventSubscriber] | None = None,
        sanitizer: ToolResultSanitizer | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._config = config
        self._context_providers = list(context_providers or [])
        self._tool_sources = list(tool_sources or [])
        self._post_step_hooks = list(post_step_hooks or [])
        self._event_subscribers = list(event_subscribers or [])
        # sanitizer 为 None 时 Engine 行为与加入该字段前完全一致（向后兼容）。
        # 非 None 时对 untrusted 工具结果做 prompt injection 防御，并在 system
        # prompt 里注入 security guard section 告知 LLM 围栏语义。
        self._sanitizer = sanitizer
        self._cancel_event = asyncio.Event()
        # subscriber 失败计数（按 name 分组）。critical=True 的 subscriber 失败
        # 应该被外部健康检查消费，决定是否标记实例为 degraded。
        self.subscriber_failures: dict[str, int] = {}

    def cancel(self) -> None:
        self._cancel_event.set()

    def reset_cancel(self) -> None:
        self._cancel_event.clear()

    def tool_source_names(self) -> list[str]:
        """已注册的 ToolSource 名字列表。H-A3 公共访问器，替代私有字段读取。"""
        return [
            getattr(s, "name", type(s).__name__) for s in self._tool_sources
        ]

    def tool_names(self) -> list[str]:
        """已注册的静态 ToolSpec 名字（不含 ToolSource 动态工具）。"""
        return [t.name for t in self._tools]

    def add_event_subscriber(self, subscriber: EventSubscriber) -> None:
        """追加一个 EventSubscriber。Engine 构造后的能力装配（如 metrics）走此接口。"""
        self._event_subscribers.append(subscriber)

    def capabilities_report(self) -> dict[str, list[str]]:
        """一站式能力快照：工具 / 工具源 / 上下文提供者 / 订阅者名字。
        调用方（如 Agent.from_config 返回值、browser_agent 校验）用它验证能力装配。
        """
        return {
            "tools": self.tool_names(),
            "tool_sources": self.tool_source_names(),
            "context_providers": [
                getattr(p, "name", type(p).__name__)
                for p in self._context_providers
            ],
            "event_subscribers": [
                getattr(s, "name", type(s).__name__)
                for s in self._event_subscribers
            ],
            "post_step_hooks": [
                getattr(h, "name", type(h).__name__)
                for h in self._post_step_hooks
            ],
        }

    def _raise_if_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise Cancelled()

    async def _emit(self, event: Event) -> None:
        for subscriber in self._event_subscribers:
            try:
                await subscriber.on_event(event)
            except Exception as exc:
                sub_name = getattr(
                    subscriber, "name", type(subscriber).__name__
                )
                is_critical = bool(getattr(subscriber, "critical", False))
                self.subscriber_failures[sub_name] = (
                    self.subscriber_failures.get(sub_name, 0) + 1
                )
                log_level = logging.ERROR if is_critical else logging.WARNING
                _logger.log(
                    log_level,
                    "event subscriber %r failed on %s: %r%s",
                    sub_name,
                    event.type.value,
                    exc,
                    " [CRITICAL]" if is_critical else "",
                )

    async def _snapshot_tools(self) -> list[ToolSpec]:
        tools = list(self._tools)
        seen = {tool.name for tool in tools}
        for source in self._tool_sources:
            self._raise_if_cancelled()
            # 每一步都重新拉取动态工具，保证 MCP 一类的外部工具列表是最新快照。
            dynamic = await source.list_tools()
            for tool in dynamic:
                if tool.name in seen:
                    continue
                seen.add(tool.name)
                tools.append(tool)
        return tools

    async def _collect_ephemeral_context(self, session: Session) -> list[Message]:
        collected: list[Message] = []
        for provider in self._context_providers:
            self._raise_if_cancelled()
            collected.extend(await provider.provide(session))
        return collected

    def _build_call_messages(
        self, session: Session, ephemeral: list[Message]
    ) -> list[Message]:
        builder = PromptBuilder()

        # session.system_prompt 是最高优先级的 section
        if session.system_prompt:
            builder.add("system-prompt", session.system_prompt, SectionPriority.SYSTEM_PROMPT)

        # sanitizer 开启时注入 security guard，解释 <tool_output> 围栏语义给 LLM。
        # 放在 INSTRUCTIONS 优先级附近，确保处于 system prompt 较显著位置。
        if self._sanitizer is not None:
            builder.add(
                SECURITY_GUARD_TAG,
                SECURITY_GUARD_CONTENT,
                SectionPriority.INSTRUCTIONS,
            )

        non_system: list[Message] = []
        for msg in ephemeral:
            if msg.role == Role.SYSTEM and msg.content:
                # 从 Message.extra 中读取 section 元信息，无则使用默认值
                tag = msg.extra.get("section_tag", "context") if msg.extra else "context"
                priority = msg.extra.get("section_priority", 500) if msg.extra else 500
                builder.add(tag, msg.content, priority)
            else:
                non_system.append(msg)

        result: list[Message] = []
        system_text = builder.build()
        if system_text:
            result.append(Message(role=Role.SYSTEM, content=system_text))
        result.extend(non_system)
        result.extend(session.messages)
        return result

    @staticmethod
    def _find_tool(name: str, pool: list[ToolSpec]) -> ToolSpec | None:
        for tool in pool:
            if tool.name == name:
                return tool
        return None

    def _transition(self, session: Session, state: RunState) -> Event:
        session.state = state
        return Event(
            type=EventType.STATE_CHANGED,
            session_id=session.id,
            payload={"state": state.value},
        )

    def _event(
        self,
        event_type: EventType,
        session: Session,
        payload: dict[str, Any],
    ) -> Event:
        return Event(type=event_type, session_id=session.id, payload=payload)

    async def _call_llm_with_cancel(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        # LLM 调用和取消信号并行等待，谁先结束就按谁的结果收口。
        llm_task = asyncio.create_task(
            self._provider.complete(
                LLMRequest(
                    model=self._config.model,
                    messages=messages,
                    tools=tools,
                    provider_options=dict(self._config.provider_options or {}),
                )
            )
        )
        cancel_task = asyncio.create_task(self._cancel_event.wait())

        done, _ = await asyncio.wait(
            {llm_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_task in done:
            llm_task.cancel()
            try:
                await llm_task
            except BaseException:
                pass
            raise Cancelled()

        cancel_task.cancel()
        try:
            await cancel_task
        except asyncio.CancelledError:
            pass
        return llm_task.result()

    async def _stream_llm_events(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        session: Session,
        step: int,
        final_holder: list[LLMResponse],
    ) -> AsyncIterator[Event]:
        """流式调用路径：yield LLM_TEXT_DELTA 事件给上层，final_holder[0] 回填最终 response。

        用 list 作为可变容器传递最终 response，规避 async generator 无法 return 值的限制。
        取消支持：每次接收一个 chunk 前检查 cancel_event，取消时 raise Cancelled()。
        """
        assert isinstance(self._provider, StreamingLLMProvider)

        request = LLMRequest(
            model=self._config.model,
            messages=messages,
            tools=tools,
            provider_options=dict(self._config.provider_options or {}),
        )

        stream = self._provider.stream(request)
        final: LLMResponse | None = None
        try:
            async for chunk in stream:
                if self._cancel_event.is_set():
                    raise Cancelled()

                if chunk.type == "text_delta" and chunk.text_delta:
                    yield self._event(
                        EventType.LLM_TEXT_DELTA,
                        session,
                        {"step": step, "delta": chunk.text_delta},
                    )
                elif chunk.type == "done" and chunk.final_response is not None:
                    final = chunk.final_response
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

        if final is None:
            # Provider 没发 done chunk，当作异常返回空响应，避免引擎卡住
            final = LLMResponse(
                text="",
                tool_calls=[],
                finish_reason="error",
                usage={},
                response_metadata=None,
            )
        final_holder.append(final)

    async def _run_post_step_hooks(self, session: Session, step: int) -> None:
        for hook in self._post_step_hooks:
            self._raise_if_cancelled()
            await hook.after_step(session, step)

    async def run(self, session: Session) -> AsyncIterator[Event]:
        # run 负责包住完整生命周期，统一发出 RUN_START / RUN_END。
        run_start = Event(
            type=EventType.RUN_START,
            session_id=session.id,
            payload={
                "model": self._config.model,
                "goal": session.goal,
                "initial_message_count": len(session.messages),
                "max_steps": self._config.max_steps,
            },
        )
        await self._emit(run_start)
        yield run_start

        final_state = session.state.value
        async for event in self._run_inner(session):
            await self._emit(event)
            yield event
            if event.type == EventType.STATE_CHANGED:
                final_state = event.payload.get("state", final_state)

        run_end = Event(
            type=EventType.RUN_END,
            session_id=session.id,
            payload={
                "final_state": final_state,
                "message_count": len(session.messages),
            },
        )
        await self._emit(run_end)
        yield run_end

    async def _run_inner(self, session: Session) -> AsyncIterator[Event]:
        yield self._transition(session, RunState.RUNNING)

        try:
            for step in range(self._config.max_steps):
                self._raise_if_cancelled()
                yield self._event(EventType.STEP_START, session, {"step": step})

                # 先冻结本步用到的上下文和工具集，避免一步内前后视图不一致。
                ephemeral = await self._collect_ephemeral_context(session)
                tools_snapshot = await self._snapshot_tools()
                call_messages = self._build_call_messages(session, ephemeral)

                use_stream = (
                    self._config.stream
                    and isinstance(self._provider, StreamingLLMProvider)
                )
                yield self._event(
                    EventType.LLM_CALL_START,
                    session,
                    {
                        "step": step,
                        "model": self._config.model,
                        "tool_count": len(tools_snapshot),
                        "ephemeral_msg_count": len(ephemeral),
                        "call_msg_count": len(call_messages),
                        "stream": use_stream,
                    },
                )
                if use_stream:
                    final_holder: list[LLMResponse] = []
                    async for evt in self._stream_llm_events(
                        call_messages, tools_snapshot, session, step, final_holder,
                    ):
                        yield evt
                    response = final_holder[0]
                else:
                    response = await self._call_llm_with_cancel(
                        call_messages, tools_snapshot
                    )
                yield self._event(
                    EventType.LLM_CALL_END,
                    session,
                    {
                        "step": step,
                        "tool_call_count": len(response.tool_calls),
                        "finish_reason": response.finish_reason,
                        "usage": response.usage,
                    },
                )

                # H-R2 token budget 计费与强制
                _accumulate_usage(session, response.usage)
                if (
                    session.token_budget is not None
                    and session.token_spent > session.token_budget
                ):
                    raise BudgetExceeded(
                        f"session {session.id} token budget exceeded: "
                        f"{session.token_spent} > {session.token_budget}"
                    )

                self._raise_if_cancelled()

                # 无论后面是否要调工具，模型这一步的 assistant 输出都先落到会话里。
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response.text,
                    tool_calls=list(response.tool_calls),
                    extra=wrap_response_metadata(response.response_metadata),
                )
                session.messages.append(assistant_msg)
                yield self._event(
                    EventType.MESSAGE_APPENDED,
                    session,
                    {
                        "role": Role.ASSISTANT.value,
                        "tool_call_count": len(response.tool_calls),
                    },
                )

                if not response.tool_calls:
                    # 没有工具调用就说明本轮推理结束，直接进入完成态。
                    await self._run_post_step_hooks(session, step)
                    yield self._transition(session, RunState.DONE)
                    return

                async for event in self._execute_tool_calls(
                    session, response.tool_calls, tools_snapshot
                ):
                    yield event

                await self._run_post_step_hooks(session, step)
                yield self._event(EventType.STEP_END, session, {"step": step})

            yield self._event(
                EventType.STEP_END,
                session,
                {"reason": "max_steps_reached"},
            )
            yield self._transition(session, RunState.DONE)

        except Cancelled:
            yield self._event(EventType.CANCELLED, session, {})
            yield self._transition(session, RunState.WAITING_USER)
        except Exception as exc:
            yield self._event(
                EventType.ERROR,
                session,
                {"kind": type(exc).__name__, "message": str(exc)},
            )
            yield self._transition(session, RunState.ERROR)

    async def _execute_tool_calls(
        self,
        session: Session,
        calls: list[ToolCall],
        pool: list[ToolSpec],
    ) -> AsyncIterator[Event]:
        for call in calls:
            self._raise_if_cancelled()

            tool = self._find_tool(call.name, pool)
            yield self._event(
                EventType.TOOL_CALL_START,
                session,
                {
                    "name": call.name,
                    "call_id": call.id,
                    "registered": tool is not None,
                },
            )

            if tool is None:
                result = ToolResult(
                    call_id=call.id,
                    output=f"tool '{call.name}' not registered",
                    is_error=True,
                )
                trust_level = "trusted"
            else:
                trust_level = getattr(tool, "trust_level", "trusted")
                try:
                    # 工具上下文把取消信号和调用元信息一起传给 handler。
                    ctx = ToolContext(
                        session_id=session.id,
                        call_id=call.id,
                        cancel_event=self._cancel_event,
                    )
                    output = await tool.handler(call.arguments, ctx)
                    result = ToolResult(call_id=call.id, output=output)
                except Cancelled:
                    raise
                except Exception as exc:
                    result = ToolResult(
                        call_id=call.id,
                        output=f"{type(exc).__name__}: {exc}",
                        is_error=True,
                    )

            # Prompt injection 防御：untrusted 工具结果在落入 session.messages 前消毒。
            # sanitizer 为 None 时直通，保证向后兼容。
            if self._sanitizer is not None:
                try:
                    result = self._sanitizer.sanitize(result, trust_level=trust_level)
                except Exception:
                    _logger.warning(
                        "sanitizer failed for tool %s, passing through",
                        call.name,
                        extra={
                            "session_id": session.id,
                            "call_id": call.id,
                            "tool_name": call.name,
                        },
                        exc_info=True,
                    )

            # 工具结果也写回会话，供下一轮 LLM 继续读取和推理。
            session.messages.append(Message(role=Role.TOOL, tool_results=[result]))
            yield self._event(
                EventType.MESSAGE_APPENDED,
                session,
                {"role": Role.TOOL.value, "call_id": call.id},
            )
            yield self._event(
                EventType.TOOL_CALL_END,
                session,
                {
                    "name": call.name,
                    "call_id": call.id,
                    "is_error": result.is_error,
                },
            )
