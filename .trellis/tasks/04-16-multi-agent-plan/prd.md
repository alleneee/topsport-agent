# Multi-Agent Plan Mode

## Goal

Implement a plan-based execution mode where an orchestrator agent generates a DAG
of steps, the user reviews and approves the plan, then sub-agents execute the steps
in parallel (respecting dependency order).

Reference: Claude Code's plan mode (plan first, approve, then execute).

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sub-agent isolation | Fully isolated (own Engine + Session) | Zero Engine changes, clean failure boundaries |
| Plan structure | DAG with explicit dependencies | Steps without deps auto-parallel via topological sort |
| User approval | Full plan review -> approve -> execute, pause on step failure | Claude Code pattern |

## Requirements

### Plan Data Model
- `Plan` with ordered list of `PlanStep`
- Each `PlanStep`: id, title, instructions, depends_on (list of step ids), status
- Step statuses: pending, running, done, failed, skipped
- DAG validation: no cycles, all dependency ids exist

### Plan Generation
- Orchestrator uses existing Engine to generate a plan via LLM
- LLM returns structured plan (tool call or structured output)
- Plan is presented to user for review
- User can approve, reject, or request edits

### DAG Execution
- Topological sort to determine execution waves
- Steps with all deps satisfied run in parallel (asyncio.gather)
- Each step spawns an isolated Engine + Session
- Step failure policy: pause and wait for user decision (retry / skip / abort)

### Orchestrator
- New module: `engine/orchestrator.py`
- Takes a `Plan` + sub-agent factory config
- Yields events during execution (plan-level + forwarded sub-agent events)
- Respects parent Engine's cancel signal

### Sub-Agent Factory
- Creates independent Engine + Session per step
- Configurable: model, tools, context providers per step
- Step instructions become the sub-agent's goal/system prompt

## Acceptance Criteria

- [ ] `Plan` and `PlanStep` dataclasses with DAG validation
- [ ] Plan generation via LLM tool call
- [ ] User approval flow (approve / reject)
- [ ] DAG executor with parallel step execution
- [ ] Step failure pauses execution, waits for user decision
- [ ] Cancel propagation from parent to all running sub-agents
- [ ] Events emitted for plan lifecycle (plan.created, plan.approved, step.start, step.end, plan.done)
- [ ] All tests pass without optional dependencies
- [ ] Full test coverage for DAG validation, execution ordering, failure handling

## Non-Goals

- Persistent plan storage (plans live in memory only)
- Plan editing after approval (v2)
- Dynamic re-planning during execution (v2)
- Sub-agent communication / shared state (by design: fully isolated)

## Technical Notes

- `RunState.WAITING_CONFIRM` already exists -- use for plan approval pause
- New `EventType` values needed for plan lifecycle events
- Orchestrator is NOT a subclass of Engine; it composes Engine instances
- Sub-agent tools/providers configurable per step, default to parent's config
