"""Capability module protocol — uniform assembly primitive for Agent features.

Before this module: Agent.from_config was a god factory — each capability
(skills / memory / plugins / browser / file_ops) had its own hand-rolled
if-branch with its own shape of side effects (tools vs context_providers vs
tool_sources vs cleanup). Adding a new capability meant editing the factory
and remembering which list to append to.

After: each capability is a `CapabilityModule` — one method `install(ctx)`
that returns a `CapabilityBundle`. The factory just asks each registered
module "are you enabled?" and merges their bundles in order. Adding a new
capability = one new module class, zero factory changes.

Dependency ordering: modules execute in their registered order. A module
that publishes state (e.g. PluginsModule publishes `plugin_manager`) must
be registered before any module that consumes it via `ctx.shared`. For now
the ordering is a simple list in `_default_capability_modules()`; if the
module count grows past a handful, swap to explicit `depends_on` + topo-sort.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..engine.hooks import (
        ContextProvider,
        EventSubscriber,
        PostStepHook,
        ToolSource,
    )
    from ..llm.provider import LLMProvider
    from ..types.tool import ToolSpec
    from .base import Agent, AgentConfig


@dataclass(slots=True)
class InstallContext:
    """What a capability module receives at install time.

    `shared` is populated by earlier modules; later modules read it. For
    example PluginsModule writes `plugin_manager` into shared, then
    SkillsModule reads it to pull extra skill_dirs.

    `parent_ref` is a single-element-or-empty list used for late binding:
    spawn_executor inside PluginsModule needs a reference to the fully
    constructed Agent, but Agent isn't built until AFTER all modules
    install. The factory appends the Agent into parent_ref once construction
    finishes, and closures captured inside modules dereference lazily.
    """

    config: "AgentConfig"
    provider: "LLMProvider"
    shared: dict[str, Any] = field(default_factory=dict)
    parent_ref: list["Agent"] = field(default_factory=list)


@dataclass(slots=True)
class CapabilityBundle:
    """What a capability module contributes to the Agent assembly.

    Every field is optional — a module that only adds tools leaves everything
    else empty. Merger semantics are "extend list / update dict", no dedup:
    collision handling (e.g. tool name conflict) is owned by the engine's
    snapshot/dedup logic, not by the bundle merge.
    """

    tools: list["ToolSpec"] = field(default_factory=list)
    context_providers: list["ContextProvider"] = field(default_factory=list)
    tool_sources: list["ToolSource"] = field(default_factory=list)
    post_step_hooks: list["PostStepHook"] = field(default_factory=list)
    event_subscribers: list["EventSubscriber"] = field(default_factory=list)
    cleanup_callbacks: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    # state is both "published to later modules" (written into ctx.shared after
    # install) AND "exposed to Agent constructor" (keys like skill_registry,
    # plugin_manager are lifted into Agent attributes for direct access).
    state: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CapabilityModule(Protocol):
    """Uniform contract for Agent capability assembly.

    Implementations should be:
      - stateless (they hold no instance state across install calls; all
        runtime state lives in the bundle they return)
      - side-effect-free before install() (constructing the module must not
        touch disk / network)
      - idempotent up to external resources (a second install() call on the
        same context may re-create disk layouts but shouldn't double-wire
        hooks on the same Agent)
    """

    @property
    def name(self) -> str: ...

    def is_enabled(self, ctx: InstallContext) -> bool: ...

    def install(self, ctx: InstallContext) -> CapabilityBundle: ...
