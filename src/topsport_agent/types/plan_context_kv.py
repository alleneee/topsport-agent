"""Reusable key-value shared context for Plan execution.

Provides KVPlanContext, a single-field PlanContext subclass with a merge-style
reducer that lets steps publish arbitrary key-value pairs:

    plan_context_merge(key="kv", value={"step1_result": "..."})
    plan_context_read()  # -> {"kv": {"step1_result": "..."}}

When caller need typed channels (specific field names with declared reducers),
define a custom PlanContext subclass directly. KVPlanContext is the generic
"shared scratch pad" option, used by the HTTP server and available to any
entry point that wants the same shape.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from .plan_context import PlanContext, Reducer

__all__ = ["KVPlanContext", "dict_merge_reducer"]


def dict_merge_reducer(current: dict[str, Any], update: Any) -> dict[str, Any]:
    """Dict-merge reducer: new keys overwrite, others preserved.

    Tolerates JSON-string input: some OpenAI/Anthropic-compatible endpoints
    (e.g. MiniMax) JSON-stringify nested object values in tool_use.input
    rather than passing real objects. Detect a str-encoded dict and decode
    it once before merging.
    """
    if isinstance(update, str):
        try:
            update = json.loads(update)
        except ValueError as exc:
            raise TypeError(
                f"value was a string and could not be JSON-decoded into an object: {exc}"
            ) from exc
    if not isinstance(update, dict):
        raise TypeError(
            f"value must be an object/dict, got {type(update).__name__}"
        )
    return {**current, **update}


class KVPlanContext(PlanContext):
    """Generic PlanContext with a single 'kv' dict field and merge-reducer.

    Usage (from Agent / HTTP / CLI any entry point):

        plan = Plan(..., context=KVPlanContext())

    LLM-facing tools auto-mounted by Orchestrator when context is set:
      - plan_context_read()                                → {"kv": {...}}
      - plan_context_merge(key="kv", value={"a": 1, "b": 2})
    """

    kv: Annotated[dict[str, Any], Reducer(dict_merge_reducer)] = {}
