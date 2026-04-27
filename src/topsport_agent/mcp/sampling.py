"""MCP `sampling` capability — server-initiated LLM calls.

Per the MCP spec (2025-11-25):
    - Client declares `capabilities.sampling = {...}` at init.
    - Server requests LLM completions via `sampling/createMessage` with
      messages / model preferences / sampling params.
    - Client responds with the LLM's reply (or an error).

The point: agent servers can ask "summarise this log paragraph",
"classify this user input", etc. **without needing their own model
credentials** — they reuse the agent's `LLMProvider`.

This module provides:
    - `SamplingMessage` / `SamplingRequest` / `SamplingResult`: stable
      dataclass façades so callers don't need the `mcp` SDK installed
      to declare a handler.
    - `SamplingHandler`: Protocol implementations resolve the request
      and return a result. The default `LLMProviderSamplingHandler`
      bridges to a `LLMProvider`, picks a model from preferences, caps
      tokens, and converts back.
    - Lazy SDK conversion helpers (`from_sdk_params` /
      `to_sdk_result`) that import `mcp.types` only at the SDK boundary.

Why a façade dataclass instead of `mcp.types.CreateMessageRequestParams`:
    - `mcp` extra is optional; tests / handlers shouldn't depend on it.
    - Server-driven sampling has security implications (the client
      lends its LLM credit to whoever it talks to). A stable, narrow
      surface is easier to audit than the raw SDK type.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..llm.provider import LLMProvider

_logger = logging.getLogger(__name__)

SamplingRole = Literal["user", "assistant"]


@dataclass(slots=True, frozen=True)
class SamplingMessage:
    """A single message in a server-driven sampling request.

    `content` is plain text; the SDK's image/audio variants are not
    surfaced here yet (matches the agent's current LLMRequest shape —
    multimodal sampling can be added in a follow-up Phase).
    """

    role: SamplingRole
    content: str


@dataclass(slots=True, frozen=True)
class SamplingRequest:
    """Server-driven LLM call as a stable client-side type.

    Fields mirror MCP `CreateMessageRequestParams` but are flat / Python-
    natural:
      - `messages`: ordered conversation
      - `system_prompt`: optional override
      - `model_hints`: ordered list of model names the server prefers
        (best-effort match; `LLMProviderSamplingHandler` picks the
        first available)
      - `cost_priority` / `speed_priority` / `intelligence_priority`:
        0..1 floats per spec; default handler ignores (provider-aware
        handler may use them)
      - `max_tokens` / `temperature` / `stop_sequences`: standard
    """

    messages: list[SamplingMessage]
    system_prompt: str | None = None
    model_hints: list[str] = field(default_factory=list)
    cost_priority: float | None = None
    speed_priority: float | None = None
    intelligence_priority: float | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class SamplingResult:
    """Reply to a SamplingRequest.

    `model` is the actual model identifier the handler used (may differ
    from any hint when no hint matched — operators can audit).
    `stop_reason` mirrors MCP enum: "endTurn" / "stopSequence" /
    "maxTokens" / etc.
    """

    role: SamplingRole
    content: str
    model: str
    stop_reason: str | None = None


SamplingHandler = Callable[[SamplingRequest], Awaitable[SamplingResult]]


# ---------------------------------------------------------------------------
# Rate limiting (P1 review): bound the cost of server-driven LLM calls
# ---------------------------------------------------------------------------


class RateLimitExceeded(Exception):
    """Raised by `RateLimitStrategy.check` to signal that the next sampling
    call would exceed the configured budget. `LLMProviderSamplingHandler`
    catches this from `__call__` and re-raises so `MCPClient._sampling_callback`
    turns it into a JSON-RPC -32603 error visible to the server."""


@runtime_checkable
class RateLimitStrategy(Protocol):
    """Pluggable rate-limit policy for server-driven sampling.

    Implementations decide algorithm (token bucket / sliding window /
    leaky bucket / fixed window) and dimensions (per-client / per-model
    / global). The simple default `TokenBucketRateLimit` ships below;
    operators with stricter / more granular requirements (per-tenant,
    cost-aware) implement this Protocol.
    """

    async def check(self, client_name: str) -> None:
        """Called BEFORE each sampling LLM call. Raise `RateLimitExceeded`
        to short-circuit; return None to allow."""
        ...


class TokenBucketRateLimit:
    """Simple per-handler token bucket: `capacity` calls available, refilled
    at `refill_per_minute` rate. `client_name` parameter is currently logged
    but not used for sharding — the bucket is per-handler-instance. To
    isolate buckets per client, give each client its own
    `LLMProviderSamplingHandler(rate_limit=TokenBucketRateLimit(...))`.

    Default values (you decided these in PR review):
      capacity=60 calls, refill_per_minute=60 → ~1 call/sec sustained,
      burst up to 60 from cold start.
    """

    def __init__(
        self,
        *,
        capacity: int = 60,
        refill_per_minute: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if refill_per_minute <= 0:
            raise ValueError(
                f"refill_per_minute must be > 0, got {refill_per_minute}"
            )
        self._capacity = capacity
        self._refill_rate = refill_per_minute / 60.0  # tokens/sec
        self._clock = clock
        self._tokens: float = float(capacity)  # start full
        self._last_refill: float = clock()
        self._lock = asyncio.Lock()

    async def check(self, client_name: str) -> None:
        async with self._lock:
            now = self._clock()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity, self._tokens + elapsed * self._refill_rate,
            )
            self._last_refill = now
            if self._tokens < 1:
                raise RateLimitExceeded(
                    f"sampling rate limit exceeded for client={client_name!r}: "
                    f"{self._capacity} cap, {self._refill_rate * 60:.0f}/min refill"
                )
            self._tokens -= 1


# ---------------------------------------------------------------------------
# Default handler: bridge to LLMProvider
# ---------------------------------------------------------------------------


class LLMProviderSamplingHandler:
    """Default sampling handler: routes server requests to the agent's
    own LLMProvider.

    Picks the first hint that matches `allowed_models` (when set);
    otherwise falls back to `default_model`. Caps `max_tokens` at
    `max_tokens_cap` (operators set this to bound cost on
    server-initiated calls — server can theoretically request large
    completions, so we always have a hard ceiling). Logs at INFO when
    server requests sampling so operators see the reverse-channel
    activity in the regular log pipeline.
    """

    def __init__(
        self,
        provider: "LLMProvider",
        *,
        default_model: str,
        allowed_models: list[str] | None = None,
        max_tokens_cap: int = 4096,
        rate_limit: RateLimitStrategy | None = None,
        client_name: str = "?",
    ) -> None:
        if not default_model:
            raise ValueError(
                "LLMProviderSamplingHandler: default_model required"
            )
        if max_tokens_cap <= 0:
            raise ValueError(
                f"max_tokens_cap must be > 0, got {max_tokens_cap}"
            )
        self._provider = provider
        self._default_model = default_model
        self._allowed_models = list(allowed_models) if allowed_models else None
        self._max_tokens_cap = max_tokens_cap
        self._rate_limit = rate_limit
        # client_name 仅作 audit 日志标识，缺省 "?" 时 audit 仍能 emit 但不便定位；
        # 应用层（lifespan）应在每个 client 装配时显式传它的 name。
        self._client_name = client_name

    def _select_model(self, hints: list[str]) -> str:
        """Pick a model from hints subject to the allowlist, or fall
        back to default. Returns a model identifier the provider can
        consume."""
        for hint in hints:
            if self._allowed_models is None or hint in self._allowed_models:
                return hint
        return self._default_model

    async def __call__(self, request: SamplingRequest) -> SamplingResult:
        from ..llm.request import LLMRequest
        from ..types.message import Message, Role

        # Rate-limit gate (P1 from review): operators wire a strategy via
        # `rate_limit` constructor arg. None = unlimited (default for
        # backward-compat; production should set one). Strategy raises
        # `RateLimitExceeded` to short-circuit; _sampling_callback turns
        # that into JSON-RPC -32603 the server can backoff against.
        if self._rate_limit is not None:
            await self._rate_limit.check(self._client_name)

        model = self._select_model(request.model_hints)

        # Cap tokens: prefer the smaller of server's request and our cap.
        max_out = (
            min(request.max_tokens, self._max_tokens_cap)
            if request.max_tokens is not None
            else self._max_tokens_cap
        )

        # Convert SamplingMessage -> internal Message
        messages: list[Message] = []
        if request.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=request.system_prompt))
        for m in request.messages:
            role = Role.USER if m.role == "user" else Role.ASSISTANT
            messages.append(Message(role=role, content=m.content))

        # P2 from review: stop_sequences currently dropped because
        # LLMRequest doesn't have a field for it yet. Warn so operators
        # see it; follow-up to thread it through to anthropic / openai
        # adapters when LLMRequest gains the slot.
        if request.stop_sequences:
            _logger.warning(
                "mcp.sampling: stop_sequences=%r dropped (LLMRequest lacks field; "
                "follow-up wiring planned)",
                request.stop_sequences,
            )

        llm_req = LLMRequest(
            model=model,
            messages=messages,
            tools=[],
            max_output_tokens=max_out,
            temperature=request.temperature,
        )

        # Audit log (P1): client_name + model + tokens — enough for
        # post-hoc forensic correlation when bills go weird.
        _logger.info(
            "mcp.sampling: client=%s model=%s max_tokens=%s msg_count=%d",
            self._client_name, model, max_out, len(request.messages),
        )

        response = await self._provider.complete(llm_req)
        return SamplingResult(
            role="assistant",
            content=response.text or "",
            model=model,
            stop_reason=response.finish_reason,
        )


# ---------------------------------------------------------------------------
# SDK boundary: convert MCP types <-> our dataclasses
# ---------------------------------------------------------------------------


def from_sdk_params(params: Any) -> SamplingRequest:
    """Convert MCP SDK `CreateMessageRequestParams` → `SamplingRequest`.

    Image / audio content blocks raise ValueError — current handler
    contract is text-only. Follow-up multimodal support keyed on this
    function.
    """
    msgs: list[SamplingMessage] = []
    for sdk_msg in getattr(params, "messages", None) or []:
        role_raw = getattr(sdk_msg, "role", "user")
        role: SamplingRole = "user" if role_raw == "user" else "assistant"
        content = getattr(sdk_msg, "content", None)
        # mcp SDK content is a single TextContent / ImageContent / AudioContent
        text = getattr(content, "text", None) if content is not None else None
        if text is None:
            raise ValueError(
                "mcp.sampling: non-text content blocks not supported yet"
            )
        msgs.append(SamplingMessage(role=role, content=text))

    prefs = getattr(params, "modelPreferences", None)
    hints: list[str] = []
    cost = speed = intel = None
    if prefs is not None:
        for h in getattr(prefs, "hints", None) or []:
            name = getattr(h, "name", None)
            if name:
                hints.append(name)
        cost = getattr(prefs, "costPriority", None)
        speed = getattr(prefs, "speedPriority", None)
        intel = getattr(prefs, "intelligencePriority", None)

    return SamplingRequest(
        messages=msgs,
        system_prompt=getattr(params, "systemPrompt", None),
        model_hints=hints,
        cost_priority=cost,
        speed_priority=speed,
        intelligence_priority=intel,
        max_tokens=getattr(params, "maxTokens", None),
        temperature=getattr(params, "temperature", None),
        stop_sequences=list(getattr(params, "stopSequences", None) or []),
    )


def to_sdk_result(result: SamplingResult) -> Any:
    """Convert `SamplingResult` → MCP SDK `CreateMessageResult`. Lazy
    import so handler code paths don't pull the SDK when unused."""
    import importlib

    mcp_types = importlib.import_module("mcp.types")
    text_content = mcp_types.TextContent(type="text", text=result.content)
    return mcp_types.CreateMessageResult(
        role=result.role,
        content=text_content,
        model=result.model,
        stopReason=result.stop_reason,
    )


def to_sdk_error(message: str, code: int = -32603) -> Any:
    """Build an `ErrorData` for the SDK to send back when the handler
    fails. Default code -32603 (Internal error) per JSON-RPC spec."""
    import importlib

    mcp_types = importlib.import_module("mcp.types")
    return mcp_types.ErrorData(code=code, message=message)


__all__ = [
    "LLMProviderSamplingHandler",
    "RateLimitExceeded",
    "RateLimitStrategy",
    "SamplingHandler",
    "SamplingMessage",
    "SamplingRequest",
    "SamplingResult",
    "SamplingRole",
    "TokenBucketRateLimit",
    "from_sdk_params",
    "to_sdk_error",
    "to_sdk_result",
]
