"""Capability module protocol — uniform assembly primitive for Agent features.

Before this module: Agent.from_config was a god factory — each capability
(skills / memory / plugins / browser / file_ops) had its own hand-rolled
if-branch with its own shape of side effects (tools vs context_providers vs
tool_sources vs cleanup). Adding a new capability meant editing the factory
and remembering which list to append to.

After: each capability is a `CapabilityModule` — one method `install(ctx)`
that returns a `CapabilityBundle`. The factory just asks each registered
module "are you enabled?" and merges their bundles in dependency order.
Adding a new capability = one new module class, zero factory changes.

Dependency ordering: each module declares its prerequisites via
`depends_on: tuple[str, ...]` (names of other modules that must publish
state into `ctx.shared` before this one runs). `Agent.from_config` runs a
`graphlib.TopologicalSorter` over the registered set, falling back to
registration order for ties. Missing or cyclic dependencies fail fast
with `CapabilityWiringError` rather than silently shipping a broken Agent.
"""

from __future__ import annotations

import graphlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..engine.hooks import (
        ContextProvider,
        EventSubscriber,
        PostStepHook,
        PostToolUseHook,
        PreToolUseHook,
        ToolSource,
    )
    from ..engine.permission.audit import AuditLogger
    from ..engine.permission.filter import ToolVisibilityFilter
    from ..engine.sanitizer import ToolResultSanitizer
    from ..llm.provider import LLMProvider
    from ..types.permission import PermissionAsker, PermissionChecker
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

    List fields are accumulated across modules (extend, no dedup). Scalar
    fields (sanitizer / permission_*) are typically populated once from
    AgentConfig in `Agent.from_config`; modules that produce them use
    last-non-None semantics via `merge()`.

    This dataclass is the single source of truth for the parent capability
    surface that `Agent.spawn_child` hands down to sub-agents. Previously the
    same payload lived as `dict[str, Any]`, which let typo'd keys silently
    drop subscribers/tracing on spawn paths (see commit f02235a).
    """

    tools: list["ToolSpec"] = field(default_factory=list)
    context_providers: list["ContextProvider"] = field(default_factory=list)
    tool_sources: list["ToolSource"] = field(default_factory=list)
    post_step_hooks: list["PostStepHook"] = field(default_factory=list)
    event_subscribers: list["EventSubscriber"] = field(default_factory=list)
    pre_tool_hooks: list["PreToolUseHook"] = field(default_factory=list)
    post_tool_hooks: list["PostToolUseHook"] = field(default_factory=list)
    cleanup_callbacks: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    sanitizer: "ToolResultSanitizer | None" = None
    permission_filter: "ToolVisibilityFilter | None" = None
    audit_logger: "AuditLogger | None" = None
    permission_checker: "PermissionChecker | None" = None
    permission_asker: "PermissionAsker | None" = None
    # state is both "published to later modules" (written into ctx.shared after
    # install) AND "exposed to Agent constructor" (keys like skill_registry,
    # plugin_manager are lifted into Agent attributes for direct access).
    state: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "CapabilityBundle") -> None:
        """Fold `other` into self in place. Lists extend; scalars take other
        when non-None; state dict updates."""
        self.tools.extend(other.tools)
        self.context_providers.extend(other.context_providers)
        self.tool_sources.extend(other.tool_sources)
        self.post_step_hooks.extend(other.post_step_hooks)
        self.event_subscribers.extend(other.event_subscribers)
        self.pre_tool_hooks.extend(other.pre_tool_hooks)
        self.post_tool_hooks.extend(other.post_tool_hooks)
        self.cleanup_callbacks.extend(other.cleanup_callbacks)
        if other.sanitizer is not None:
            self.sanitizer = other.sanitizer
        if other.permission_filter is not None:
            self.permission_filter = other.permission_filter
        if other.audit_logger is not None:
            self.audit_logger = other.audit_logger
        if other.permission_checker is not None:
            self.permission_checker = other.permission_checker
        if other.permission_asker is not None:
            self.permission_asker = other.permission_asker
        self.state.update(other.state)


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

    Dependency declaration:
      `depends_on` lists names of other modules that must run first because
      this module reads `ctx.shared[<key>]` written by them. Implementations
      must declare it (use empty `()` for no dependencies) — `name` and
      `depends_on` are class-level attributes; Protocol attribute form here
      stays compatible with the `name = "..."` / `depends_on = ()` style
      used by every in-tree module under capability_impls.
    """

    name: str
    depends_on: tuple[str, ...]

    def is_enabled(self, ctx: InstallContext) -> bool: ...

    def install(self, ctx: InstallContext) -> CapabilityBundle: ...


class CapabilityWiringError(RuntimeError):
    """Raised when registered CapabilityModules cannot be ordered.

    Two failure modes:
      - Unknown dependency: a module declares depends_on=("foo",) but no
        registered module has name "foo".
      - Cycle: depends_on edges form a directed cycle.

    Both are configuration bugs; `from_config` fails fast so the operator
    sees the misconfiguration at startup rather than at first tool call.
    """


def order_capability_modules(
    modules: Sequence[CapabilityModule],
) -> list[CapabilityModule]:
    """Topologically sort modules by `depends_on`, breaking ties on the
    original registration order.

    Tie-break rule: when several modules are simultaneously ready (no
    remaining unmet deps), they are emitted in the order they appear in
    `modules`. This keeps the user-visible install order stable as long
    as no new dependency edges are introduced.

    Raises `CapabilityWiringError` on unknown dependency names or cycles.
    Empty input returns []; a single module returns [module] regardless
    of declared deps (only checked against the registered set).
    """
    if not modules:
        return []

    by_name: dict[str, CapabilityModule] = {}
    registration_index: dict[str, int] = {}
    for idx, mod in enumerate(modules):
        name = mod.name
        if name in by_name:
            existing_idx = registration_index[name]
            raise CapabilityWiringError(
                f"duplicate CapabilityModule name {name!r}: "
                f"already registered at position {existing_idx}, "
                f"cannot register again at position {idx}. Common cause: "
                f"`AgentConfig.extra_capability_modules` contains a module "
                f"whose `name` collides with a default module."
            )
        by_name[name] = mod
        registration_index[name] = idx

    sorter: graphlib.TopologicalSorter[str] = graphlib.TopologicalSorter()
    for mod in modules:
        # depends_on is required by the Protocol; missing attribute treated as
        # `()` so legacy / third-party duck-typed modules don't break, but
        # explicit declaration is preferred (see Protocol docstring).
        deps = tuple(getattr(mod, "depends_on", ()) or ())
        if mod.name in deps:
            raise CapabilityWiringError(
                f"CapabilityModule {mod.name!r} declares a self-dependency "
                f"(depends_on contains its own name)"
            )
        for dep in deps:
            if dep not in by_name:
                raise CapabilityWiringError(
                    f"CapabilityModule {mod.name!r} depends on {dep!r}, "
                    f"but no module with that name is registered"
                )
        sorter.add(mod.name, *deps)

    try:
        sorter.prepare()
    except graphlib.CycleError as exc:
        # graphlib.CycleError.args[1] holds the cycle path when present;
        # keep a defensive fallback that surfaces the raw args so the
        # operator still sees which modules are involved instead of an
        # opaque "unknown" placeholder.
        if len(exc.args) > 1 and isinstance(exc.args[1], (list, tuple)):
            cycle_repr = " -> ".join(str(n) for n in exc.args[1])
        else:
            cycle_repr = repr(exc.args)
        raise CapabilityWiringError(
            f"CapabilityModule dependency cycle detected: {cycle_repr}"
        ) from exc

    ordered: list[CapabilityModule] = []
    while sorter.is_active():
        ready = list(sorter.get_ready())
        # Stable tie-break: among simultaneously-ready modules, emit in
        # the order the operator originally registered them.
        ready.sort(key=lambda n: registration_index[n])
        for name in ready:
            ordered.append(by_name[name])
            sorter.done(name)
    return ordered
