from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..llm.provider import LLMProvider, LLMResponse, StreamingLLMProvider
from ..llm.request import LLMRequest
from ..llm.response import wrap_response_metadata
from ..tools.blob_store import BlobStore
from ..tools.output_cap import enforce_cap
from ..types.events import Event, EventType
from ..types.message import Message, Role, ToolCall, ToolResult
from ..types.session import RunState, Session
from ..types.tool import ToolContext, ToolSpec
from .hooks import (
    ContextProvider,
    EventSubscriber,
    HookAllow,
    HookDeny,
    PostStepHook,
    PostToolUseHook,
    PreToolUseHook,
    ToolSource,
)
from .prompt import PromptBuilder, SectionPriority
from .sanitizer import SECURITY_GUARD_CONTENT, SECURITY_GUARD_TAG, ToolResultSanitizer

if TYPE_CHECKING:
    from ..types.permission import PermissionAsker, PermissionChecker
    from .permission.audit import AuditLogger
    from .permission.filter import ToolVisibilityFilter

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


@dataclass(slots=True, frozen=True)
class EngineRunOptions:
    max_output_tokens: int | None = None
    temperature: float | None = None
    provider_options: dict[str, Any] | None = None


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
        pre_tool_hooks: list[PreToolUseHook] | None = None,
        post_tool_hooks: list[PostToolUseHook] | None = None,
        sanitizer: ToolResultSanitizer | None = None,
        blob_store: BlobStore | None = None,
        default_max_result_chars: int | None = None,
        permission_checker: "PermissionChecker | None" = None,
        permission_asker: "PermissionAsker | None" = None,
        permission_filter: "ToolVisibilityFilter | None" = None,
        audit_logger: "AuditLogger | None" = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._config = config
        self._context_providers = list(context_providers or [])
        self._tool_sources = list(tool_sources or [])
        self._post_step_hooks = list(post_step_hooks or [])
        self._event_subscribers = list(event_subscribers or [])
        # PreToolUseHook / PostToolUseHook 链：tool 调用前/后的扩展点，对标 Claude
        # Code 的 PreToolUse / PostToolUse。permission_checker / sanitizer 仍走专属
        # 路径（permission 多阶段 ASK 决策、sanitizer 围栏注入）；hooks 是“我也想插
        # 一脚”的通用入口。空列表完全等价于关闭，零额外开销。
        self._pre_tool_hooks = list(pre_tool_hooks or [])
        self._post_tool_hooks = list(post_tool_hooks or [])
        # sanitizer 为 None 时 Engine 行为与加入该字段前完全一致（向后兼容）。
        # 非 None 时对 untrusted 工具结果做 prompt injection 防御，并在 system
        # prompt 里注入 security guard section 告知 LLM 围栏语义。
        self._sanitizer = sanitizer
        # blob_store 为 None 时 Engine 不做 output cap（或仅切片但不落盘）——向后兼容。
        # 非 None 时，工具结果超过 ToolSpec.max_result_chars（或 default_max_result_chars）
        # 自动落盘，返回 {preview, blob_ref, original_size} 给 LLM，避免 context 爆炸。
        self._blob_store = blob_store
        self._default_max_result_chars = default_max_result_chars
        # permission_checker 为 None 时引擎行为与以前完全一致（不做任何权限检查）。
        # 注入 DefaultPermissionChecker 可启用基于 ToolSpec 字段的默认策略；
        # 具体 asker 由调用方实现（CLI 终端 / server SSE / CI 环境等）。
        self._permission_checker = permission_checker
        self._permission_asker = permission_asker
        # v2 capability-ACL hot-path hooks. permission_filter 为 None 时 _snapshot_tools
        # 直通（向后兼容）；非 None 时在每步快照末尾做 required_permissions ⊆ granted
        # 过滤 + 可选 kill-switch。audit_logger 为 None 时 _invoke_tool 不产生 AuditEntry。
        self._permission_filter = permission_filter
        self._audit_logger = audit_logger
        # v2 hot path：_run_inner 开始时把 session 注入到这里，供 _snapshot_tools
        # 默认参数回退使用。外部 patch 0-arg spy 也能正确读取当前 session。
        self._current_session: Session | None = None
        self._cancel_event = asyncio.Event()
        # PreToolUseHook 链没有并发安全契约（hook 可能 prompt 用户、读写共享状态、
        # 限流/审计依赖顺序）。当 LLM 一次发多个 concurrency_safe 工具调用、
        # _invoke_tool 被并发调度时，用这把锁把 hook 链强制串行化，保留"按注册
        # 顺序执行"的语义；handler 仍可并发。post hook 走在 _execute_tool_calls
        # 的串行 for-loop 内，天然不受影响。
        self._pre_tool_hook_lock = asyncio.Lock()
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

    def add_tool_source(self, source: ToolSource) -> None:
        """追加一个 ToolSource。Engine 构造后按需注入动态工具（如 orchestrator 的
        plan_context bridge）。每步 _snapshot_tools 会看到新源。"""
        self._tool_sources.append(source)

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

    async def _snapshot_tools(self, session: Session | None = None) -> list[ToolSpec]:
        # session 参数在 v2 capability-ACL 后被引擎内部通过 self._current_session
        # 注入（见 _run_inner 的 try/finally 段）；显式传参也能覆盖，便于上层直接调用。
        # 默认 None 是为了让老测试 / 外部 patch 无需关心签名差异。
        if session is None:
            session = self._current_session
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
        # v2 capability-ACL：静态能力过滤 + kill-switch 由 filter 统一处理。
        # 未注入 filter 时完全跳过，保持旧行为（granted_permissions 不强制）。
        if self._permission_filter is not None and session is not None:
            tools = await self._permission_filter.filter(tools, session)
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

    def _build_llm_request(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        run_options: EngineRunOptions | None,
    ) -> LLMRequest:
        provider_options = dict(self._config.provider_options or {})
        overrides = run_options.provider_options if run_options else None
        for key, value in (overrides or {}).items():
            if isinstance(value, dict) and isinstance(provider_options.get(key), dict):
                merged = dict(provider_options[key])
                merged.update(value)
                provider_options[key] = merged
            else:
                provider_options[key] = value
        return LLMRequest(
            model=self._config.model,
            messages=messages,
            tools=tools,
            max_output_tokens=run_options.max_output_tokens if run_options else None,
            temperature=run_options.temperature if run_options else None,
            provider_options=provider_options,
        )

    async def _call_llm_with_cancel(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        run_options: EngineRunOptions | None,
    ) -> LLMResponse:
        # LLM 调用和取消信号并行等待，谁先结束就按谁的结果收口。
        llm_task = asyncio.create_task(
            self._provider.complete(self._build_llm_request(messages, tools, run_options))
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
        run_options: EngineRunOptions | None,
    ) -> AsyncIterator[Event]:
        """流式调用路径：yield LLM_TEXT_DELTA 事件给上层，final_holder[0] 回填最终 response。

        用 list 作为可变容器传递最终 response，规避 async generator 无法 return 值的限制。
        取消支持：每次接收一个 chunk 前检查 cancel_event，取消时 raise Cancelled()。
        """
        assert isinstance(self._provider, StreamingLLMProvider)

        request = self._build_llm_request(messages, tools, run_options)

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

    async def run(
        self,
        session: Session,
        *,
        run_options: EngineRunOptions | None = None,
    ) -> AsyncIterator[Event]:
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
        async for event in self._run_inner(session, run_options):
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

    async def _run_inner(
        self,
        session: Session,
        run_options: EngineRunOptions | None,
    ) -> AsyncIterator[Event]:
        # 把 session 挂到 self，确保 _snapshot_tools / 其他 hot-path 方法在被外部
        # 0-arg 测试替身覆盖时仍能看到当前 session（capability filter 需要 session
        # 的 granted_permissions / tenant_id）。run 结束后 finally 中清零避免跨会话泄漏。
        self._current_session = session
        try:
            yield self._transition(session, RunState.RUNNING)

            try:
                for step in range(self._config.max_steps):
                    self._raise_if_cancelled()
                    yield self._event(EventType.STEP_START, session, {"step": step})

                    # 先冻结本步用到的上下文和工具集，避免一步内前后视图不一致。
                    ephemeral = await self._collect_ephemeral_context(session)
                    # 不传 session 位置参数：允许测试替身以 0-arg spy 覆盖本方法。
                    # 默认参数分支从 self._current_session 读取当前 session。
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
                            call_messages,
                            tools_snapshot,
                            session,
                            step,
                            final_holder,
                            run_options,
                        ):
                            yield evt
                        response = final_holder[0]
                    else:
                        response = await self._call_llm_with_cancel(
                            call_messages, tools_snapshot, run_options
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
        finally:
            self._current_session = None

    async def _invoke_tool(
        self,
        call: ToolCall,
        tool: ToolSpec | None,
        session: Session,
    ) -> tuple[ToolResult, str, dict[str, Any]]:
        """执行单个 tool_call：validate_input → hook → permission → handler。

        返回 (result, trust_level, effective_args)：effective_args 是经过
        PreToolUseHook / permission 改写后真正喂给 handler 的 args，下游
        PostToolUseHook 用它构造 hook_call，避免观察到陈旧 args。
        独立成方法是为了让并发组（concurrency_safe）可预先 asyncio.create_task 本方法，
        外层 yield 事件时 await 已完成的 task，事件顺序保持和 calls 列表一致。
        """
        if tool is None:
            result = ToolResult(
                call_id=call.id,
                output=f"tool '{call.name}' not registered",
                is_error=True,
            )
            await self._audit_call(session, tool, call.arguments, result)
            return result, "trusted", call.arguments
        trust_level = getattr(tool, "trust_level", "trusted")
        # effective_args 提前初始化以便所有早返回点都能携带它给调用方
        # （PostToolUseHook 用它构造看见最终 args 的 hook_call）。
        effective_args = call.arguments
        # Pre-flight 参数校验：返回错误字符串则跳过 handler，直接回 LLM 自我修正。
        # 对标 CC 的 validateInput() -> ValidationResult。
        validator = getattr(tool, "validate_input", None)
        if validator is not None:
            try:
                err = await validator(call.arguments)
            except Cancelled:
                raise
            except Exception as exc:
                result = ToolResult(
                    call_id=call.id,
                    output=f"validate_input raised {type(exc).__name__}: {exc}",
                    is_error=True,
                )
                await self._audit_call(session, tool, call.arguments, result)
                return result, trust_level, effective_args
            if err is not None:
                result = ToolResult(
                    call_id=call.id, output=err, is_error=True,
                )
                await self._audit_call(session, tool, call.arguments, result)
                return result, trust_level, effective_args

        ctx = ToolContext(
            session_id=session.id,
            call_id=call.id,
            cancel_event=self._cancel_event,
            # workspace_root is the file_ops sandbox boundary. None keeps
            # CLI trust mode (no restriction); server populates session.workspace
            # at creation so HTTP-initiated tool calls can't escape to host FS.
            workspace_root=(
                session.workspace.files_dir if session.workspace is not None else None
            ),
        )

        # 改写跟踪：仅当 PreToolUseHook 或 permission_checker 显式改写过 args
        # 时才需要在 handler 调用前再跑一次 validate_input。
        args_rewritten = False

        # PreToolUseHook 链：放在 permission_checker **之前**，使 permission 始终
        # 对最终 args 做把关（hook 改写后 permission 重新评估）。Hook 看到的是
        # effective_args 的 deepcopy：嵌套对象的 in-place 突变也改不到上游，
        # 想改写必须走 HookAllow(updated_args=...)，由 args_rewritten flag 保证
        # 后续 revalidate。Hook 抛异常视为透传（log warning，不阻塞工具调用）。
        # 整段在 _pre_tool_hook_lock 内执行：即便 _invoke_tool 被多个 concurrent
        # tool calls 并发调度，hook 链跨调用之间也是串行的（防 hook 共享状态竞争）。
        if self._pre_tool_hooks:
            async with self._pre_tool_hook_lock:
                for hook in self._pre_tool_hooks:
                    pre_call = ToolCall(
                        id=call.id,
                        name=call.name,
                        arguments=copy.deepcopy(effective_args),
                    )
                    try:
                        decision_h = await hook.before_tool(pre_call, tool, ctx)
                    except Cancelled:
                        raise
                    except Exception:
                        _logger.warning(
                            "pre_tool_hook %r raised; passing through for %s",
                            getattr(hook, "name", type(hook).__name__),
                            call.name,
                            exc_info=True,
                        )
                        continue
                    if isinstance(decision_h, HookDeny):
                        result = ToolResult(
                            call_id=call.id,
                            output=decision_h.reason or f"tool '{call.name}' denied by hook",
                            is_error=True,
                        )
                        await self._audit_call(session, tool, effective_args, result)
                        return result, trust_level, effective_args
                    if isinstance(decision_h, HookAllow) and decision_h.updated_args is not None:
                        effective_args = decision_h.updated_args
                        args_rewritten = True

        # 改写后立即重跑 validate_input：避免 hook 写出非法 args 直接进入
        # permission_checker 与 handler。permission_checker 用最终 args 决策。
        if validator is not None and args_rewritten:
            try:
                err = await validator(effective_args)
            except Cancelled:
                raise
            except Exception as exc:
                result = ToolResult(
                    call_id=call.id,
                    output=f"validate_input (post-hook) raised {type(exc).__name__}: {exc}",
                    is_error=True,
                )
                await self._audit_call(session, tool, effective_args, result)
                return result, trust_level, effective_args
            if err is not None:
                result = ToolResult(
                    call_id=call.id,
                    output=f"validate_input (post-hook): {err}",
                    is_error=True,
                )
                await self._audit_call(session, tool, effective_args, result)
                return result, trust_level, effective_args
            # validator 通过后清 flag；permission rewrite 自己再 set 一次
            args_rewritten = False

        # Permission check：checker 返回 DENY/ASK→(asker→)→ 最终决策。
        # checker 为 None 完全跳过（兼容现有行为）。当 hook 已改写过 args，
        # 用改写后的版本构造 check_call，让 permission 看到最终值。
        if self._permission_checker is not None:
            check_call = (
                call if effective_args is call.arguments
                else ToolCall(id=call.id, name=call.name, arguments=effective_args)
            )
            try:
                decision = await self._permission_checker.check(tool, check_call, ctx)
            except Cancelled:
                raise
            except Exception as exc:
                _logger.warning(
                    "permission_checker %r raised %r for %s; treating as DENY",
                    getattr(self._permission_checker, "name", "?"),
                    exc,
                    call.name,
                    exc_info=True,
                )
                result = ToolResult(
                    call_id=call.id,
                    output=f"permission_checker error: {type(exc).__name__}: {exc}",
                    is_error=True,
                )
                await self._audit_call(session, tool, effective_args, result)
                return result, trust_level, effective_args
            # 用字符串字面量比较 behavior，避免在模块级导入 legacy PermissionBehavior
            # 触发自己的 DeprecationWarning。v1 runtime-decision 路径整体下个版本下线。
            if decision.behavior == "ask":
                if self._permission_asker is None:
                    # 无 asker 的保守默认：直接拒绝。日志里记一下，便于排查。
                    _logger.info(
                        "no PermissionAsker configured; denying %s (reason=%r)",
                        call.name, decision.reason,
                    )
                    result = ToolResult(
                        call_id=call.id,
                        output=decision.reason or "permission ask without asker; denied",
                        is_error=True,
                    )
                    await self._audit_call(session, tool, effective_args, result)
                    return result, trust_level, effective_args
                try:
                    decision = await self._permission_asker.ask(
                        tool, check_call, ctx, decision.reason,
                    )
                except Cancelled:
                    raise
                except Exception as exc:
                    _logger.warning(
                        "permission_asker %r raised %r for %s; treating as DENY",
                        getattr(self._permission_asker, "name", "?"),
                        exc,
                        call.name,
                        exc_info=True,
                    )
                    result = ToolResult(
                        call_id=call.id,
                        output=f"permission_asker error: {type(exc).__name__}: {exc}",
                        is_error=True,
                    )
                    await self._audit_call(session, tool, effective_args, result)
                    return result, trust_level, effective_args
                # asker 再返回 ASK 是契约违反，保守按 DENY 处理
                if decision.behavior == "ask":
                    import warnings as _w

                    with _w.catch_warnings():
                        _w.simplefilter("ignore", DeprecationWarning)
                        from ..types.permission import (
                            PermissionBehavior as _PB,
                            PermissionDecision as _PD,
                        )
                    decision = _PD(
                        _PB.DENY,
                        reason="asker returned ASK; contract violation",
                    )
            if decision.behavior == "deny":
                result = ToolResult(
                    call_id=call.id,
                    output=decision.reason or f"tool '{call.name}' denied",
                    is_error=True,
                )
                await self._audit_call(session, tool, effective_args, result)
                return result, trust_level, effective_args
            # ALLOW：允许 checker/asker 改写入参（如安全路径重写）
            if decision.updated_input is not None:
                effective_args = decision.updated_input
                args_rewritten = True

        # permission 路径若再次改写 args，再跑一次 validator 兜底。
        if validator is not None and args_rewritten:
            try:
                err = await validator(effective_args)
            except Cancelled:
                raise
            except Exception as exc:
                result = ToolResult(
                    call_id=call.id,
                    output=f"validate_input (post-permission) raised {type(exc).__name__}: {exc}",
                    is_error=True,
                )
                await self._audit_call(session, tool, effective_args, result)
                return result, trust_level, effective_args
            if err is not None:
                result = ToolResult(
                    call_id=call.id,
                    output=f"validate_input (post-rewrite): {err}",
                    is_error=True,
                )
                await self._audit_call(session, tool, effective_args, result)
                return result, trust_level, effective_args

        try:
            output = await tool.handler(effective_args, ctx)
            # 自动 blob offload：超过 ToolSpec.max_result_chars 时全量落盘 + 预览回传。
            # 无 blob_store 或 cap 未设置时 enforce_cap 行为退化（仅切片或直通）。
            cap = tool.max_result_chars if tool.max_result_chars is not None else self._default_max_result_chars
            if cap is not None:
                cap_result = enforce_cap(output, cap, self._blob_store)
                output = cap_result.output
            result = ToolResult(call_id=call.id, output=output)
            await self._audit_call(session, tool, effective_args, result)
            return result, trust_level, effective_args
        except Cancelled:
            raise
        except Exception as exc:
            result = ToolResult(
                call_id=call.id,
                output=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
            await self._audit_call(session, tool, effective_args, result)
            return result, trust_level, effective_args

    async def _audit_call(
        self,
        session: Session,
        tool: ToolSpec | None,
        args: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Emit audit entry for a completed tool call. Swallows logger errors.

        outcome="error" on ToolResult.is_error else "allowed". reason 仅在错误路径
        记录原始 output 文本，便于排查；args_preview 由 AuditLogger 内部 redact。
        """
        if self._audit_logger is None:
            return
        outcome = "error" if result.is_error else "allowed"
        reason = str(result.output) if result.is_error else None
        try:
            await self._audit_logger.log_call(
                session=session, tool=tool, args=args,
                outcome=outcome, reason=reason,
            )
        except Exception:
            _logger.warning("audit log_call failed", exc_info=True)

    async def _execute_tool_calls(
        self,
        session: Session,
        calls: list[ToolCall],
        pool: list[ToolSpec],
    ) -> AsyncIterator[Event]:
        # 并发分组策略：concurrency_safe 的 handler 预先 create_task 后台跑，事件仍按
        # calls 原顺序 yield（测试可预期）。unsafe 的走原地串行。连续多个 read_only
        # browser_get_text / search 这种一次能省大量 wall-clock。
        scheduled: dict[int, asyncio.Task[tuple[ToolResult, str, dict[str, Any]]]] = {}
        for idx, call in enumerate(calls):
            tool = self._find_tool(call.name, pool)
            if tool is not None and getattr(tool, "concurrency_safe", False):
                scheduled[idx] = asyncio.create_task(
                    self._invoke_tool(call, tool, session)
                )

        for idx, call in enumerate(calls):
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

            if idx in scheduled:
                # 并发已调度——要么已完成、要么很快完成；await 拿结果即可。
                try:
                    result, trust_level, eff_args = await scheduled[idx]
                except Cancelled:
                    # 取消其他未完成的并发 task，避免资源泄漏
                    for other_idx, task in scheduled.items():
                        if other_idx != idx and not task.done():
                            task.cancel()
                    raise
            else:
                result, trust_level, eff_args = await self._invoke_tool(call, tool, session)

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

            # PostToolUseHook 链：sanitizer 后、append 到 messages 前。
            # 链式 result -> hook1 -> hook2 -> ...；hook 抛异常视为透传。
            # hook 看到的 call.arguments 应是 handler 实际收到的 effective_args
            # （PreToolUseHook 改写 / permission rewrite 之后的最终值），不是
            # LLM 提交的原 args，否则审计/redaction 类 hook 会观察到陈旧输入。
            if self._post_tool_hooks:
                tool_for_hook = self._find_tool(call.name, pool)
                hook_ctx = ToolContext(
                    session_id=session.id,
                    call_id=call.id,
                    cancel_event=self._cancel_event,
                    workspace_root=(
                        session.workspace.files_dir
                        if session.workspace is not None else None
                    ),
                )
                hook_call = (
                    call if eff_args is call.arguments
                    else ToolCall(id=call.id, name=call.name, arguments=eff_args)
                )
                for hook in self._post_tool_hooks:
                    try:
                        result = await hook.after_tool(
                            hook_call, tool_for_hook, result, hook_ctx,
                            trust_level=trust_level,
                        )
                    except Cancelled:
                        raise
                    except Exception:
                        _logger.warning(
                            "post_tool_hook %r raised; passing previous result for %s",
                            getattr(hook, "name", type(hook).__name__),
                            call.name,
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
