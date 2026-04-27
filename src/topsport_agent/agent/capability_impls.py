"""Default CapabilityModule implementations for in-tree features.

Each class here is a one-file migration of a branch that previously lived
inline in `Agent.from_config`. Adding a new capability should follow the
same template: implement the protocol, declare `depends_on`, register in
default_capability_modules. The factory uses `order_capability_modules`
for topo-sort, so module order in the registry list is no longer
load-bearing — only `depends_on` edges matter.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .capabilities import CapabilityBundle, InstallContext

if TYPE_CHECKING:
    from .base import Agent


def _async_wrap(func: Callable[..., Any]) -> Callable[[], Awaitable[None]]:
    """Wrap a sync cleanup function as an async-callable, mirroring the pattern
    used by the old inline factory for plugin_manager.cleanup."""

    async def _call() -> None:
        func()

    return _call


# ---------------------------------------------------------------------------
# Plugins — must run first; publishes `plugin_manager` + `spawn_parent_getter`
# ---------------------------------------------------------------------------


class PluginsModule:
    name = "plugins"
    depends_on: tuple[str, ...] = ()

    def is_enabled(self, ctx: InstallContext) -> bool:
        return ctx.config.enable_plugins

    def install(self, ctx: InstallContext) -> CapabilityBundle:
        from ..plugins import PluginManager
        from ..plugins.agent_registry import build_agent_tools
        from .base import _build_spawn_executor  # avoid circular at module load

        manager = PluginManager()
        manager.load()

        # spawn_agent needs the fully-constructed parent Agent; capture the
        # late-binding slot here and defer dereference to actual tool call.
        parent_ref = ctx.parent_ref

        def _parent_getter() -> Agent:
            if not parent_ref:
                raise RuntimeError(
                    "spawn_agent invoked before parent Agent finished construction"
                )
            return parent_ref[0]

        executor = _build_spawn_executor(_parent_getter)
        bundle = CapabilityBundle()
        bundle.tools.extend(build_agent_tools(manager.agent_registry(), executor))
        bundle.event_subscribers.append(manager.hook_runner())
        bundle.cleanup_callbacks.append(_async_wrap(manager.cleanup))
        bundle.state["plugin_manager"] = manager
        return bundle


# ---------------------------------------------------------------------------
# Skills — depends on plugin_manager (consumes skill_dirs from plugins)
# ---------------------------------------------------------------------------


class SkillsModule:
    name = "skills"
    # 读 ctx.shared["plugin_manager"] 以收集 plugin 提供的 skill_dirs，因此必须
    # 在 PluginsModule 之后跑；显式声明依赖，topo-sort 自动安排顺序。
    depends_on: tuple[str, ...] = ("plugins",)

    def is_enabled(self, ctx: InstallContext) -> bool:
        return ctx.config.enable_skills

    def install(self, ctx: InstallContext) -> CapabilityBundle:
        from ..skills import (
            SkillInjector,
            SkillLoader,
            SkillMatcher,
            SkillRegistry,
            build_skill_tools,
        )

        plugin_manager = ctx.shared.get("plugin_manager")
        plugin_skill_dirs = plugin_manager.skill_dirs() if plugin_manager else []
        all_skill_dirs = list(ctx.config.local_skill_dirs) + plugin_skill_dirs

        registry = SkillRegistry(all_skill_dirs)
        registry.load()
        loader = SkillLoader(registry)
        matcher = SkillMatcher(registry)

        bundle = CapabilityBundle()
        bundle.tools.extend(build_skill_tools(registry, matcher))
        bundle.context_providers.append(SkillInjector(registry, loader, matcher))
        bundle.state["skill_registry"] = registry
        return bundle


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryModule:
    name = "memory"
    depends_on: tuple[str, ...] = ()

    def is_enabled(self, ctx: InstallContext) -> bool:
        return ctx.config.enable_memory

    def install(self, ctx: InstallContext) -> CapabilityBundle:
        from ..memory.file_store import FileMemoryStore
        from ..memory.injector import MemoryInjector
        from ..memory.tools import build_memory_tools

        base = ctx.config.memory_base_path or (
            Path.home() / ".topsport-agent" / "memory"
        )
        store = FileMemoryStore(base)

        bundle = CapabilityBundle()
        bundle.tools.extend(build_memory_tools(store))
        bundle.context_providers.append(MemoryInjector(store))
        return bundle


# ---------------------------------------------------------------------------
# File ops (read/write/edit/list/glob/grep) — now driven by
# AgentConfig.enable_file_ops for consistency with other flags. Previously
# this was mixed in via default_agent's extra_tools.
# ---------------------------------------------------------------------------


class FileOpsModule:
    name = "file_ops"
    depends_on: tuple[str, ...] = ()

    def is_enabled(self, ctx: InstallContext) -> bool:
        return ctx.config.enable_file_ops

    def install(self, ctx: InstallContext) -> CapabilityBundle:
        from ..tools import file_tools

        del ctx  # file_ops doesn't need any context state currently
        bundle = CapabilityBundle()
        bundle.tools.extend(file_tools())
        return bundle


# ---------------------------------------------------------------------------
# Browser — Playwright-backed; silently skips when playwright isn't available
# ---------------------------------------------------------------------------


class BrowserModule:
    name = "browser"
    depends_on: tuple[str, ...] = ()

    def is_enabled(self, ctx: InstallContext) -> bool:
        return ctx.config.enable_browser

    def install(self, ctx: InstallContext) -> CapabilityBundle:
        from .base import _try_make_browser

        del ctx
        bundle = CapabilityBundle()
        browser_client, browser_sources = _try_make_browser()
        if browser_client is not None:
            bundle.tool_sources.extend(browser_sources)
            bundle.cleanup_callbacks.append(browser_client.close)
        return bundle


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def default_capability_modules() -> list[Any]:
    """Default capability modules registered for `Agent.from_config`.

    The list order is **not** load-bearing for execution: `from_config`
    runs `order_capability_modules()` to topo-sort by `depends_on`,
    breaking ties on the order below. To make a new module run before
    another, declare `depends_on` explicitly rather than reordering this
    list. Tie-break order kept stable for backward-compatible behavior
    when no new edges are introduced.
    """
    return [
        PluginsModule(),
        SkillsModule(),
        MemoryModule(),
        FileOpsModule(),
        BrowserModule(),
    ]
