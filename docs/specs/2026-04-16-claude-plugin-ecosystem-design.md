# Claude Plugin Ecosystem Integration

Load the full Claude Code plugin ecosystem (`~/.claude/plugins/`) into
topsport-agent, supporting all four extension types: Skills, Commands,
Agents, and Hooks.

## New Module: `src/topsport_agent/plugins/`

```
plugins/
    __init__.py
    discovery.py          # read installed_plugins.json -> InstalledPlugin list
    plugin.py             # scan_plugin() -> PluginDescriptor
    manager.py            # PluginManager: unified entry point
    agent_registry.py     # AgentDefinition + AgentRegistry + build_agent_tools()
    hook_runner.py        # PluginHookRunner (EventSubscriber impl)
```

## 1. Plugin Discovery (`discovery.py`)

- Read `~/.claude/plugins/installed_plugins.json`
- Key format: `"name@marketplace"` -> take last entry (latest version)
- Skip entries where `installPath` does not exist
- Return `list[InstalledPlugin]` with name, marketplace, install_path, version
- `plugins_dir` is injectable for testing

## 2. Plugin Scanning (`plugin.py`)

- `PluginDescriptor` holds: info + skill_dirs + command_paths + agent_paths + hooks_config
- `scan_plugin(installed)` walks the plugin root to classify paths:
  - `skills/*/SKILL.md` -> skill_dirs
  - `commands/*.md` -> command_paths
  - `agents/*.md` -> agent_paths
  - `hooks/hooks.json` -> hooks_config

## 3. Skills & Commands Integration

- Plugin skills and commands feed into existing `SkillRegistry`
- Naming: `plugin_name:skill_name` (e.g. `superpowers:brainstorming`)
- Commands derive name from filename: `commands/brainstorm.md` -> `plugin:brainstorm`
- Commands lack `name` frontmatter; PluginManager synthesizes it before feeding to registry
- Priority: local `~/.claude/skills/` > plugin skills (local dirs listed first in SkillRegistry)
- No changes to SkillRegistry internals

## 4. Agent System (`agent_registry.py`)

### AgentDefinition

```python
@dataclass(slots=True)
class AgentDefinition:
    name: str                    # "code-reviewer"
    qualified_name: str          # "superpowers:code-reviewer"
    description: str
    body: str                    # markdown body -> sub-agent system_prompt
    allowed_tools: list[str]     # parsed from "tools:" frontmatter, empty = no restriction
    auto_skills: list[str]       # parsed from "skills:" frontmatter
    model: str                   # "inherit" | "haiku" | "sonnet" | "opus"
    source_path: Path
```

### AgentRegistry

- `load_from_plugins(plugins)` scans all `agents/*.md`
- `get(name)` / `list()` accessors
- Name collision: first plugin wins (same as SkillRegistry)

### Tools exposed to LLM

- `list_agents` -> name + description for all registered agents
- `spawn_agent(name, task)` -> creates child Engine + Session:
  - system_prompt = agent body
  - model = inherit from parent or override per agent definition
  - tools = filtered by allowed_tools (empty = inherit all)
  - auto_skills = activated in child session before run
  - Returns final assistant text output

## 5. Hook System (`hook_runner.py`)

### PluginHook

```python
@dataclass(slots=True)
class PluginHook:
    event: str            # "SessionStart", "PreToolUse", etc.
    matcher: str | None   # regex pattern, None = match all
    command: str          # shell command template
    is_async: bool        # fire-and-forget when True
    timeout: float        # seconds, default 30
    plugin_root: Path     # for ${CLAUDE_PLUGIN_ROOT} expansion
```

### Event Mapping

| Claude Hook Event  | Engine EventType                | matcher target       |
| ------------------ | ------------------------------- | -------------------- |
| SessionStart       | RUN_START                       | always               |
| PreToolUse         | TOOL_CALL_START                 | payload["name"]      |
| PostToolUse        | TOOL_CALL_END                   | payload["name"]      |
| UserPromptSubmit   | MESSAGE_APPENDED (role=user)    | always               |
| SessionEnd         | RUN_END                         | always               |

### PluginHookRunner (EventSubscriber)

- `on_event()` maps Engine event -> Claude hook event name
- Matches against registered hooks using regex matcher
- Executes command via `asyncio.create_subprocess_exec`
- Env vars injected: `CLAUDE_PLUGIN_ROOT`, `TOOL_NAME`, `SESSION_ID`
- async=true: fire-and-forget; async=false: await with timeout
- Failure: log warning, never interrupt engine
- Default timeout: 30 seconds

### from_plugins()

- Merges all plugins' hooks.json into unified hook list

## 6. PluginManager (`manager.py`)

Unified entry point:

```python
class PluginManager:
    def __init__(self, plugins_dir: Path | None = None) -> None: ...
    def load(self) -> None: ...
    def skill_dirs(self) -> list[Path]: ...
    def agent_registry(self) -> AgentRegistry: ...
    def hook_runner(self) -> PluginHookRunner: ...
```

## 7. CLI Integration (`cli/main.py`)

```python
plugin_mgr = PluginManager()
plugin_mgr.load()

local_skill_dirs = [Path.home() / ".claude" / "skills"]
all_skill_dirs = local_skill_dirs + plugin_mgr.skill_dirs()

registry = SkillRegistry(all_skill_dirs)
registry.load()
# ... wire into Engine with agent_tools + hook_runner
```

## 8. Testing

- Integration tests: scan real `~/.claude/plugins/`, verify discovery, parsing, registration
- Unit tests: temp dirs for edge cases (empty, malformed, missing files)
- No new optional dependencies
- All existing 232 tests must remain green

## Design Decisions

- SkillRegistry is unchanged; plugin integration is external wiring only
- Commands are adapted to skill format before feeding to registry
- spawn_agent is single-step (no Plan/DAG), reuses Engine directly
- Hooks are read-only/notification — cannot modify engine state
- Local skills always win over plugin skills on name collision
