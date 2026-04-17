"""Tests for Planner (LLM-based plan generation)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from topsport_agent.engine.planner import Planner, _parse_plan
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import LLMResponse
from topsport_agent.types.message import ToolCall


@dataclass
class MockProvider:
    name = "mock"
    response: LLMResponse | None = None
    seen_requests: list[LLMRequest] = field(default_factory=list)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.seen_requests.append(request)
        if self.response:
            return self.response
        return LLMResponse(text="fallback", finish_reason="stop")


def _plan_tool_call(steps: list[dict]) -> ToolCall:
    return ToolCall(id="tc1", name="create_plan", arguments={"steps": steps})


class TestPlannerGenerate:
    async def test_generates_plan_from_tool_call(self):
        provider = MockProvider(
            response=LLMResponse(
                text=None,
                tool_calls=[
                    _plan_tool_call(
                        [
                            {
                                "id": "s1",
                                "title": "First",
                                "instructions": "Do first thing",
                            },
                            {
                                "id": "s2",
                                "title": "Second",
                                "instructions": "Do second thing",
                                "depends_on": ["s1"],
                            },
                        ]
                    )
                ],
                finish_reason="tool_use",
            )
        )
        planner = Planner(provider, model="test")
        plan = await planner.generate("build something")

        assert plan.goal == "build something"
        assert len(plan.steps) == 2
        assert plan.steps[0].id == "s1"
        assert plan.steps[1].depends_on == ["s1"]

    async def test_raises_when_no_tool_call(self):
        provider = MockProvider(
            response=LLMResponse(text="I can't plan", finish_reason="stop")
        )
        planner = Planner(provider, model="test")
        with pytest.raises(ValueError, match="did not return a plan"):
            await planner.generate("something")

    async def test_raises_when_wrong_tool(self):
        provider = MockProvider(
            response=LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="tc1", name="wrong_tool", arguments={})],
                finish_reason="tool_use",
            )
        )
        planner = Planner(provider, model="test")
        with pytest.raises(ValueError, match="Expected create_plan"):
            await planner.generate("something")

    async def test_passes_context_as_message(self):
        provider = MockProvider(
            response=LLMResponse(
                text=None,
                tool_calls=[
                    _plan_tool_call(
                        [{"id": "s1", "title": "T", "instructions": "I"}]
                    )
                ],
                finish_reason="tool_use",
            )
        )
        planner = Planner(provider, model="test")
        await planner.generate("goal", context="extra context")

        req = provider.seen_requests[0]
        contents = [m.content for m in req.messages if m.content]
        assert any("extra context" in c for c in contents)

    async def test_sends_create_plan_tool_in_request(self):
        provider = MockProvider(
            response=LLMResponse(
                text=None,
                tool_calls=[
                    _plan_tool_call(
                        [{"id": "s1", "title": "T", "instructions": "I"}]
                    )
                ],
                finish_reason="tool_use",
            )
        )
        planner = Planner(provider, model="test-model")
        await planner.generate("goal")

        req = provider.seen_requests[0]
        assert req.model == "test-model"
        assert len(req.tools) == 1
        assert req.tools[0].name == "create_plan"


class TestParsePlan:
    def test_parse_valid(self):
        plan = _parse_plan(
            "goal",
            {
                "steps": [
                    {"id": "a", "title": "A", "instructions": "do A"},
                    {"id": "b", "title": "B", "instructions": "do B", "depends_on": ["a"]},
                ]
            },
        )
        assert plan.goal == "goal"
        assert len(plan.steps) == 2

    def test_parse_empty_steps_raises(self):
        with pytest.raises(ValueError, match="no steps"):
            _parse_plan("goal", {"steps": []})

    def test_parse_missing_steps_raises(self):
        with pytest.raises(ValueError, match="no steps"):
            _parse_plan("goal", {})

    def test_parse_defaults_depends_on(self):
        plan = _parse_plan(
            "goal",
            {"steps": [{"id": "a", "title": "A", "instructions": "do A"}]},
        )
        assert plan.steps[0].depends_on == []
