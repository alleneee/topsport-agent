# Permission Capability-ACL Design

- Date: 2026-04-23
- Status: Proposed
- Supersedes: the minimal `PermissionChecker/Asker` module shipped in commit `5baed29`

## 1. Context

`topsport-agent` serves as the runtime for an internal enterprise AI platform
powering productivity agents. Risky operations (shell, filesystem mutations,
arbitrary code execution) are already confined by the sandbox layer; the
permission subsystem therefore does **not** need to make runtime judgements
about specific commands.

What the platform actually needs is **capability-based access control**:
different tools (and MCP servers) have different preset permission
requirements, and each agent session carries a preset grant of capabilities.
Runtime becomes a static set-difference check — tools whose requirements are
not a subset of the session's grants are hidden from the LLM entirely.

## 2. Non-Goals

The following are explicitly out of scope because the problem shape does not
require them:

- Runtime ALLOW / DENY / ASK three-state decisions
- `PermissionAsker` interactive approval workflow
- `PendingApproval` queue / LLM-initiated `permission_request` tool
- Pattern-based rule DSL (`Bash(git *)` style matchers)
- PermissionMode (default / acceptEdits / plan / bypassPermissions)
- `.claude/settings.json` compatibility
- Hash-chain tamper-proof audit log (out of scope for internal platform)
- Dual-control approval, compliance report generation, policy canary rollout

The minimal permission module shipped in the previous commit (`Checker`,
`Asker`, `Decision`, `Behavior`) solved a problem the platform does not have.
This design replaces it.

## 3. Core Model

### 3.1 Permission

```python
class Permission(StrEnum):
    # Filesystem
    FS_READ        = "fs.read"
    FS_WRITE       = "fs.write"
    # Shell (sandbox-confined)
    SHELL_SAFE     = "shell.safe"       # ls, grep, cat
    SHELL_FULL     = "shell.full"       # rm, sudo, network-fetching shells
    # Network
    NETWORK_OUT    = "network.out"
    # MCP servers (one permission per MCP server, declared in server config)
    MCP_GITHUB     = "mcp.github"
    MCP_JIRA       = "mcp.jira"
    MCP_ZENDESK    = "mcp.zendesk"
    # Meta
    MEMORY_WRITE   = "memory.write"
    AGENT_SPAWN    = "agent.spawn"
    # ... extended as platform grows
```

Permissions are represented as strings so MCP plugins can extend the set
without code changes — `Permission` is effectively an open registry. The
`StrEnum` base is retained for IDE support on the known-ahead-of-time values;
ad-hoc permissions are passed as plain strings that satisfy the same type
contract.

### 3.2 Persona

A named bundle of permissions maintained by platform administrators:

```python
@dataclass(frozen=True, slots=True)
class Persona:
    id: str                                 # e.g. "dev_engineer"
    display_name: str
    description: str
    permissions: frozenset[Permission]
    version: int = 1
```

Example personas for an internal platform:

| Persona | Permissions |
|---|---|
| `dev_engineer` | `fs.read`, `fs.write`, `shell.full`, `network.out`, `mcp.github`, `mcp.jira` |
| `data_analyst` | `fs.read`, `network.out`, `mcp.snowflake` |
| `customer_support` | `fs.read`, `mcp.zendesk` |
| `finance_readonly` | `fs.read`, `mcp.sap` |

Personas **do not support inheritance or composition** in v1. If a set union
is needed, the administrator declares the resulting set explicitly as a new
persona. This is a deliberate YAGNI; add composition only if a concrete
internal use case justifies it.

### 3.3 PersonaAssignment

```python
@dataclass(frozen=True, slots=True)
class PersonaAssignment:
    tenant_id: str
    persona_ids: frozenset[str]
    default_persona_id: str | None
    user_id: str | None = None      # None = tenant-wide
    group_id: str | None = None     # team / department identifier
```

Resolution order when a session is created:

1. Look up assignments for `(tenant_id, user_id)` — if found, use that set
2. Else look up `(tenant_id, group_id)` — the group the user belongs to
3. Else look up `(tenant_id, None, None)` — the tenant-wide fallback
4. Else deny session creation (fail-closed)

### 3.4 AuditEntry

```python
@dataclass(frozen=True, slots=True)
class AuditEntry:
    id: str                           # UUID4
    tenant_id: str
    session_id: str
    user_id: str | None
    persona_id: str | None
    tool_name: str
    tool_required: frozenset[Permission]
    subject_granted: frozenset[Permission]
    outcome: Literal["allowed", "filtered_out", "error", "killed"]
    args_preview: dict[str, Any]      # PII-redacted, truncated to 4 KB
    reason: str | None                 # populated on error / killed
    # Reserved for future quota / cost tracking (populated as zero in v1)
    cost_tokens: int = 0
    cost_latency_ms: int = 0
    # Reserved for future group-level policy (recorded but not enforced)
    group_id: str | None = None
    timestamp: datetime
```

`AuditEntry` is `frozen=True` so event subscribers cannot mutate entries
between the logger and downstream consumers. The 4 KB cap on
`args_preview` prevents large tool payloads from bloating audit storage;
truncation is recorded as a `__truncated__: True` sentinel inside the dict.

### 3.5 ToolSpec extension

```python
@dataclass(slots=True)
class ToolSpec:
    # ...existing fields...
    required_permissions: frozenset[Permission] = frozenset()
```

The empty default preserves backwards compatibility: tools without declared
requirements are visible to any session regardless of grants. Tool authors
opt in by adding `required_permissions={Permission.FS_WRITE}` etc.

### 3.6 Session extension

```python
@dataclass(slots=True)
class Session:
    # ...existing fields (id, system_prompt, messages, state, tenant_id, ...)
    granted_permissions: frozenset[Permission] = frozenset()
    persona_id: str | None = None
```

The session owns its grants; the filter reads them per snapshot. Agent
creation copies `persona.permissions` into the session at `new_session()`
time. Grants are immutable for the lifetime of the session.

## 4. Module Structure

```
src/topsport_agent/
├── types/
│   └── permission.py                  # Full rewrite: Permission / Persona /
│                                      # PersonaAssignment / AuditEntry
├── engine/
│   └── permission/                    # Was single file; now a sub-package
│       ├── __init__.py                # Re-exports public API
│       ├── filter.py                  # ToolVisibilityFilter
│       ├── persona_registry.py        # PersonaRegistry Protocol + impls
│       ├── assignment.py              # PersonaAssignmentStore + resolver
│       ├── audit.py                   # AuditStore Protocol + AuditLogger
│       ├── redaction.py               # PIIRedactor
│       ├── metrics.py                 # PermissionMetrics (EventSubscriber)
│       └── killswitch.py              # KillSwitchGate
├── server/
│   ├── permission_api.py              # Admin / operator / auditor endpoints
│   └── rbac.py                        # FastAPI RBAC middleware
└── mcp/
    └── tool_bridge.py                 # Updated to propagate permissions
                                       # from server config into ToolSpec
```

Out-of-package dependencies:

- `engine/loop.py` imports `ToolVisibilityFilter` and `AuditLogger` from
  `engine.permission`
- `agent/base.py` imports `Persona` and `PersonaRegistry`
- `mcp/tool_bridge.py` reads `MCPServerConfig.permissions` and injects into
  bridged `ToolSpec.required_permissions`

No import cycles (server depends on engine, engine depends on types, types
depends on nothing).

## 5. Runtime Flow

### 5.1 Integration point: `Engine._snapshot_tools`

```python
async def _snapshot_tools(self, session):
    pool = list(self._tools)
    for source in self._tool_sources:
        pool.extend(await source.list_tools())
    if self._permission_filter is not None:
        pool = self._permission_filter.filter(pool, session)
    return pool
```

`ToolVisibilityFilter.filter`:

```python
def filter(
    self, pool: list[ToolSpec], session: Session
) -> list[ToolSpec]:
    if self._killswitch.is_active(session.tenant_id):
        self._audit.log_killswitch(session, pool)
        return []
    granted = session.granted_permissions
    visible: list[ToolSpec] = []
    filtered: list[ToolSpec] = []
    for tool in pool:
        if tool.required_permissions.issubset(granted):
            visible.append(tool)
        else:
            filtered.append(tool)
    if filtered:
        self._audit.log_filtered(session, filtered)
    return visible
```

The filter is called **once per step**, matching the existing
`_snapshot_tools` contract. Filtered tools never reach the LLM — the prompt
only advertises the visible subset.

### 5.2 Per-call audit

In `Engine._execute_tool_calls`, for each tool call:

```python
entry = AuditEntry(
    id=str(uuid4()),
    tenant_id=session.tenant_id,
    session_id=session.id,
    user_id=session.principal,
    persona_id=session.persona_id,
    tool_name=call.name,
    tool_required=tool.required_permissions if tool else frozenset(),
    subject_granted=session.granted_permissions,
    outcome=outcome,
    args_preview=self._redactor.redact_and_truncate(call.arguments),
    reason=reason,
    timestamp=now(),
)
await self._audit.append(entry)
```

The outcome is one of:

- `"allowed"` — tool existed, passed the filter, handler ran (regardless of
  handler success; handler errors still count as allowed)
- `"error"` — tool not registered, or filter rejected at call time (should
  not happen if filter is invoked upstream, but belt-and-braces)
- `"killed"` — kill-switch fired between filter and call (rare race)

### 5.3 MCP unified path

`mcp/config.py` extends `MCPServerConfig`:

```python
@dataclass(frozen=True)
class MCPServerConfig:
    # existing fields
    name: str
    transport: Literal["stdio", "http"]
    # ... transport-specific fields ...
    permissions: frozenset[Permission] = frozenset()
```

Sample JSON:

```json
{
  "mcpServers": {
    "github": {
      "transport": "stdio",
      "command": "mcp-server-github",
      "permissions": ["mcp.github"]
    },
    "zendesk": {
      "transport": "http",
      "url": "https://mcp.internal/zendesk",
      "permissions": ["mcp.zendesk", "network.out"]
    }
  }
}
```

`MCPToolBridge` reads `server_config.permissions` and merges them into every
bridged `ToolSpec.required_permissions`. The result is that MCP tools flow
through the exact same `ToolVisibilityFilter` pipeline as builtin tools —
one abstraction, one audit trail, one admin UI.

### 5.4 Kill-switch semantics

`KillSwitchGate` maintains a `frozenset[str]` of killed tenant IDs, backed
by the same pluggable store used for persona registry (memory / file /
eventually Redis). Toggling the switch:

1. Administrator calls `POST /v1/admin/killswitch/{tenant_id}` (role=admin)
2. Server updates the store
3. All new `filter()` calls see the kill state on the next step boundary
4. In-flight LLM calls complete normally; the next snapshot returns empty
   tool list; LLM cannot invoke further tools

There is no preemption of in-flight tool calls — those run to completion
under the existing `Engine.cancel()` semantics. Cancellation is a separate
concern and would be invoked by the administrator through the existing
session cancel path.

## 6. RBAC & Admin API

### 6.1 Roles

These roles gate **access to the administration HTTP API**. They describe
who the **HTTP caller** is (identified by the auth token), not who the
agent session's end-user is. Do not conflate with `PersonaAssignment.user_id`
in § 3.3 — that field identifies the end-user whose session runs the agent,
independent of whether that user can call the admin API.

```python
class Role(StrEnum):
    ADMIN    = "admin"      # CRUD personas, assignments, kill-switch
    OPERATOR = "operator"   # Read personas & assignments, view audit
    AUDITOR  = "auditor"    # Read-only audit query
    AGENT    = "agent"      # Runtime; no admin API access
```

### 6.2 Endpoints

| Endpoint | Method | Min Role |
|---|---|---|
| `/v1/admin/personas` | GET | OPERATOR |
| `/v1/admin/personas` | POST | ADMIN |
| `/v1/admin/personas/{id}` | PUT / DELETE | ADMIN |
| `/v1/admin/assignments` | GET | OPERATOR |
| `/v1/admin/assignments` | POST / DELETE | ADMIN |
| `/v1/admin/killswitch/{tenant_id}` | POST | ADMIN |
| `/v1/admin/killswitch/{tenant_id}` | GET | OPERATOR |
| `/v1/admin/audit` | GET | AUDITOR |

RBAC is enforced at the HTTP middleware layer (`server/rbac.py`), not inside
the engine. The engine sees pre-authorized tenant contexts.

Auth token structure (placeholder — ties to existing auth):

```python
{"sub": "alice@acme.com", "tenant_id": "acme", "role": "admin"}
```

Decoding is delegated to the existing auth layer; RBAC middleware only
reads the `role` claim and matches against the endpoint's `min_role`.

## 7. PII Redaction

`PIIRedactor` is a pattern-driven stripper applied to `args_preview` before
audit write:

```python
class PIIRedactor:
    def __init__(self, patterns: Sequence[RedactionPattern]): ...
    def redact_and_truncate(self, payload: dict[str, Any]) -> dict[str, Any]:
        # 1. deep-walk values
        # 2. apply each RedactionPattern (regex → replacement)
        # 3. json.dumps the redacted dict, truncate to 4 KB if needed
        # 4. if truncated, set payload["__truncated__"] = True
        ...
```

Default `RedactionPattern` set (extended at instantiation):

- Email: `r"[\w.+-]+@[\w-]+\.[\w.-]+"` → `"[email]"`
- Phone: internationalized digit runs → `"[phone]"`
- Credit card: Luhn-matching 13-19 digit strings → `"[cc]"`
- Token: strings matching `r"sk-[A-Za-z0-9]{20,}"` → `"[token]"`
- AWS key: `r"AKIA[0-9A-Z]{16}"` → `"[aws-key]"`

Platform operators can register additional patterns for domain-specific PII
(employee IDs, internal record numbers). Patterns are applied in declaration
order; first match wins per field.

## 8. Integration & Migration

### 8.1 Breaking changes from commit `5baed29`

The minimal permission module from the previous work is superseded:

| Symbol | Disposition |
|---|---|
| `PermissionBehavior` enum | Deprecated; removed in follow-up minor release |
| `PermissionDecision` | Deprecated; removed |
| `PermissionChecker` Protocol | Deprecated; replaced by `ToolVisibilityFilter` |
| `PermissionAsker` Protocol | Deprecated; removed |
| `DefaultPermissionChecker` | Deprecated; removed |
| `AlwaysDenyAsker` / `AlwaysAskAsker` | Deprecated; removed |
| `allow()` / `deny()` / `ask()` helpers | Deprecated; removed |
| `Engine(permission_checker=, permission_asker=)` | Parameters accepted but warn + ignore; removed in next minor |
| `tests/test_permission.py` | Replaced by `tests/test_permission_filter.py` + related |

For a transitional period the deprecated symbols live on in
`types/permission.py` behind an `__all__` marker and emit
`DeprecationWarning` on import. The transition window is one minor version.

### 8.2 New `Engine` constructor parameters

```python
Engine(
    ...
    permission_filter: ToolVisibilityFilter | None = None,
    audit_logger: AuditLogger | None = None,
    kill_switch: KillSwitchGate | None = None,
    pii_redactor: PIIRedactor | None = None,
)
```

All default to `None`, meaning pre-existing callers see no behavioural
change. When `permission_filter` is `None`, `_snapshot_tools` skips the
filter step entirely. When `audit_logger` is `None`, no audit entries are
emitted.

### 8.3 `AgentConfig` extension

```python
@dataclass
class AgentConfig:
    ...
    persona: Persona | str | None = None
    persona_registry: PersonaRegistry | None = None
    tenant_id: str | None = None
```

`Agent.from_config` resolves `persona` (string → registry lookup) and wires
`ToolVisibilityFilter` + `AuditLogger` into the engine. `Agent.new_session()`
copies `persona.permissions` into the new session.

## 9. Testing Strategy

New test files and coverage targets:

| File | Coverage |
|---|---|
| `tests/test_permission_filter.py` | Set-difference edge cases: empty grants, empty requirements, superset, disjoint, kill-switch active |
| `tests/test_persona_registry.py` | InMemory + File implementations, CRUD, version bump |
| `tests/test_persona_assignment.py` | user / group / tenant three-tier resolution |
| `tests/test_audit_store.py` | Append-only guarantee, query by tenant / time window / persona, truncation sentinel |
| `tests/test_pii_redaction.py` | Each default pattern, custom patterns, nested dict walk, 4 KB truncation |
| `tests/test_kill_switch.py` | Toggle semantics, in-flight call isolation |
| `tests/test_permission_integration_engine.py` | End-to-end: Agent with persona runs a session, verifies filter + audit |
| `tests/test_permission_mcp_unified.py` | MCP server config → bridged tool has correct `required_permissions` |
| `tests/test_permission_api.py` | FastAPI RBAC middleware, all endpoints, all role gates |

Target: zero regression on the existing 705-test suite; new permission
tests add ~50 cases.

## 10. Open Questions & Deferred Items

Deferred means "useful and likely to be added later when a concrete need
surfaces"; this is distinct from non-goals (§ 2) which are permanently out
of scope for this subsystem.

- **Redis / Postgres backends** for all stores — Protocol is ready; impl
  deferred until deployment-scale needs are clear
- **MCP connection-time gate** — skip connecting to MCP servers whose
  permissions are not in the session's grants (resource-saving
  optimization; current behaviour filters at tool-visibility stage)
- **Persona composition / inheritance** — `admin = dev_engineer ∪ finance`;
  add only when a concrete use case demands it
- **Cost tracking** — `cost_tokens` / `cost_latency_ms` fields are reserved
  but not populated
- **Group-level policy enforcement** — `group_id` is recorded in audit but
  not used for permission decisions
- **SIEM adapter** (Splunk HEC / OTLP / JSON Lines) — architecture leaves
  room; adding one is additive and does not touch the core model

## 11. Size Estimate

- Source: ~800-1000 LOC
- Tests: ~1200 LOC
- Total: ~2000 LOC
- Effort: 3-4 days to landed

---

## Appendix A: Rejected Alternatives

### A.1 Runtime ALLOW/DENY/ASK engine (the previous approach)

Rejected because the platform's risk confinement is already at the sandbox
layer. Runtime decisions would ask the wrong question ("should this bash
invocation run?") when the real question has already been answered at
agent instantiation ("is this agent allowed to bash at all?").

### A.2 Pattern-based rule DSL (`Bash(git *)` style)

Rejected because:

1. The "trigger-and-leave" workflow pattern means no one is watching to
   write or approve patterns in realtime
2. Scope-level grants cover 99% of access-control needs for an internal
   productivity platform
3. Pattern parsing / matching is per-tool infrastructure that duplicates
   the sandbox's responsibility

### A.3 CC `.claude/settings.json` compatibility

Rejected because internal enterprise deployments centralize configuration
in the platform's admin UI / database, not per-workspace JSON files. The
file format would be dead weight.

### A.4 Persona inheritance in v1

Rejected as premature optimization. Administrators can explicitly declare
union personas if needed; adding inheritance adds surface area without
concrete demand.
