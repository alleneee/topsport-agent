from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.adapters import OpenAIChatProvider
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import ProviderResponseMetadata
from topsport_agent.types.message import Message, Role, ToolCall, ToolResult
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    function: MockFunction
    type: str = "function"


@dataclass
class MockMessage:
    content: str | None = None
    tool_calls: list[MockToolCall] | None = None
    role: str = "assistant"


@dataclass
class MockChoice:
    message: MockMessage
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class MockUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class MockCompletion:
    choices: list[MockChoice]
    usage: MockUsage = field(default_factory=MockUsage)
    id: str = "chatcmpl-fake"


class MockCompletionsResource:
    def __init__(self, responses: list[MockCompletion]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> MockCompletion:
        self.requests.append(kwargs)
        if self._index >= len(self._responses):
            return MockCompletion(
                choices=[
                    MockChoice(
                        message=MockMessage(content="fallback"),
                        finish_reason="stop",
                    )
                ]
            )
        response = self._responses[self._index]
        self._index += 1
        return response


class MockChatResource:
    def __init__(self, completions: MockCompletionsResource) -> None:
        self.completions = completions


class MockOpenAIClient:
    def __init__(self, responses: list[MockCompletion]) -> None:
        self.completions = MockCompletionsResource(responses)
        self.chat = MockChatResource(self.completions)

    async def create(self, payload: dict[str, Any]) -> MockCompletion:
        return await self.completions.create(**payload)


def _text_completion(text: str, *, finish_reason: str = "stop") -> MockCompletion:
    return MockCompletion(
        choices=[
            MockChoice(
                message=MockMessage(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _tool_call_completion(
    call_id: str, name: str, arguments: dict[str, Any]
) -> MockCompletion:
    return MockCompletion(
        choices=[
            MockChoice(
                message=MockMessage(
                    content=None,
                    tool_calls=[
                        MockToolCall(
                            id=call_id,
                            function=MockFunction(
                                name=name,
                                arguments=json.dumps(arguments),
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=MockUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
    )


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> Any:
    return {"echo": args}


def _echo_tool() -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="echo back arguments",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        handler=_echo_handler,
    )


async def test_converts_simple_user_and_system_to_role_messages():
    client = MockOpenAIClient([_text_completion("hi there")])
    provider = OpenAIChatProvider(client=client)

    messages = [
        Message(role=Role.SYSTEM, content="you are helpful"),
        Message(role=Role.USER, content="hello"),
    ]
    response = await provider.complete(
        LLMRequest(model="gpt-5.1", messages=messages)
    )

    request = client.completions.requests[0]
    assert request["model"] == "gpt-5.1"
    assert request["messages"] == [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
    ]
    assert "tools" not in request
    assert response.text == "hi there"
    assert response.finish_reason == "stop"
    assert response.usage == {"input_tokens": 10, "output_tokens": 5}


async def test_system_stays_as_role_not_lifted_to_top_level():
    client = MockOpenAIClient([_text_completion("ok")])
    provider = OpenAIChatProvider(client=client)

    await provider.complete(
        LLMRequest(
            model="gpt-5.1",
            messages=[
            Message(role=Role.SYSTEM, content="rule 1"),
            Message(role=Role.SYSTEM, content="rule 2"),
            Message(role=Role.USER, content="hi"),
            ],
        )
    )

    request = client.completions.requests[0]
    assert "system" not in request
    system_msgs = [m for m in request["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 2
    assert system_msgs[0]["content"] == "rule 1"
    assert system_msgs[1]["content"] == "rule 2"


async def test_assistant_tool_calls_use_nested_function_format():
    client = MockOpenAIClient([_text_completion("ok")])
    provider = OpenAIChatProvider(client=client)

    await provider.complete(
        LLMRequest(
            model="m",
            messages=[
            Message(role=Role.USER, content="run echo"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})],
            ),
            Message(
                role=Role.TOOL,
                tool_results=[ToolResult(call_id="c1", output={"result": 1})],
            ),
            ],
        )
    )

    request_msgs = client.completions.requests[0]["messages"]
    assistant = request_msgs[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None
    assert len(assistant["tool_calls"]) == 1
    call = assistant["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"] == "c1"
    assert call["function"]["name"] == "echo"
    assert json.loads(call["function"]["arguments"]) == {"x": 1}


async def test_tool_role_messages_stay_independent_with_tool_call_id():
    client = MockOpenAIClient([_text_completion("ok")])
    provider = OpenAIChatProvider(client=client)

    await provider.complete(
        LLMRequest(
            model="m",
            messages=[
            Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={}),
                    ToolCall(id="c2", name="echo", arguments={}),
                ],
            ),
            Message(
                role=Role.TOOL,
                tool_results=[ToolResult(call_id="c1", output="first")],
            ),
            Message(
                role=Role.TOOL,
                tool_results=[ToolResult(call_id="c2", output="second")],
            ),
            ],
        )
    )

    request_msgs = client.completions.requests[0]["messages"]
    tool_msgs = [m for m in request_msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "c1"
    assert tool_msgs[0]["content"] == "first"
    assert tool_msgs[1]["tool_call_id"] == "c2"
    assert tool_msgs[1]["content"] == "second"


async def test_dict_tool_output_is_json_serialized():
    client = MockOpenAIClient([_text_completion("ok")])
    provider = OpenAIChatProvider(client=client)

    await provider.complete(
        LLMRequest(
            model="m",
            messages=[
            Message(
                role=Role.ASSISTANT,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={})],
            ),
            Message(
                role=Role.TOOL,
                tool_results=[
                    ToolResult(call_id="c1", output={"nested": {"v": 42}})
                ],
            ),
            ],
        )
    )

    tool_msg = [
        m for m in client.completions.requests[0]["messages"] if m["role"] == "tool"
    ][0]
    parsed = json.loads(tool_msg["content"])
    assert parsed == {"nested": {"v": 42}}


async def test_tool_specs_convert_to_function_type_wrapper():
    client = MockOpenAIClient([_text_completion("ok")])
    provider = OpenAIChatProvider(client=client)

    await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="hi")], tools=[_echo_tool()])
    )

    tools = client.completions.requests[0]["tools"]
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo back arguments",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        }
    ]


async def test_response_tool_calls_parse_json_arguments():
    client = MockOpenAIClient(
        [_tool_call_completion("call_abc", "echo", {"x": 5, "y": [1, 2]})]
    )
    provider = OpenAIChatProvider(client=client)

    response = await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="run echo")], tools=[_echo_tool()])
    )

    assert response.text is None
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_abc"
    assert call.name == "echo"
    assert call.arguments == {"x": 5, "y": [1, 2]}
    assert response.response_metadata == ProviderResponseMetadata(
        provider="openai",
        assistant_blocks=[
            {
                "type": "tool_use",
                "id": "call_abc",
                "name": "echo",
                "input": {"x": 5, "y": [1, 2]},
                "raw_arguments": '{"x": 5, "y": [1, 2]}',
            }
        ],
    )


async def test_malformed_json_arguments_fallback_to_raw():
    response_mock = MockCompletion(
        choices=[
            MockChoice(
                message=MockMessage(
                    content=None,
                    tool_calls=[
                        MockToolCall(
                            id="c1",
                            function=MockFunction(
                                name="echo",
                                arguments="not-json{",
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    client = MockOpenAIClient([response_mock])
    provider = OpenAIChatProvider(client=client)

    response = await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="hi")], tools=[_echo_tool()])
    )

    assert response.tool_calls[0].arguments == {"_raw_arguments": "not-json{"}


async def test_usage_fields_normalized_to_input_output_tokens():
    response_mock = _text_completion("ok")
    response_mock.usage = MockUsage(
        prompt_tokens=123, completion_tokens=45, total_tokens=168
    )
    client = MockOpenAIClient([response_mock])
    provider = OpenAIChatProvider(client=client)

    response = await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="hi")])
    )

    assert response.usage == {"input_tokens": 123, "output_tokens": 45}


async def test_reasoning_effort_from_constructor():
    client = MockOpenAIClient([_text_completion("ok")])
    provider = OpenAIChatProvider(client=client, reasoning_effort="high")

    await provider.complete(
        LLMRequest(model="gpt-5.1-thinking", messages=[Message(role=Role.USER, content="hi")])
    )

    assert client.completions.requests[0]["reasoning_effort"] == "high"


async def test_reasoning_effort_from_extra_overrides_constructor():
    client = MockOpenAIClient([_text_completion("ok")])
    provider = OpenAIChatProvider(client=client, reasoning_effort="low")

    await provider.complete(
        LLMRequest(
            model="m",
            messages=[Message(role=Role.USER, content="hi")],
            provider_options={"openai": {"reasoning_effort": "high"}},
        )
    )

    assert client.completions.requests[0]["reasoning_effort"] == "high"


async def test_max_completion_tokens_replaces_max_tokens_when_set():
    client = MockOpenAIClient([_text_completion("ok")])
    provider = OpenAIChatProvider(client=client, max_tokens=100)

    await provider.complete(
        LLMRequest(
            model="gpt-5.1-thinking",
            messages=[Message(role=Role.USER, content="hi")],
            provider_options={"openai": {"max_completion_tokens": 2048}},
        )
    )

    request = client.completions.requests[0]
    assert request["max_completion_tokens"] == 2048
    assert "max_tokens" not in request


async def test_engine_integration_with_openai_adapter():
    tool_resp = _tool_call_completion("c1", "echo", {"x": 7})
    final_resp = _text_completion("all done")
    client = MockOpenAIClient([tool_resp, final_resp])
    provider = OpenAIChatProvider(client=client)

    engine = Engine(
        provider,
        tools=[_echo_tool()],
        config=EngineConfig(model="gpt-5.1"),
    )
    session = Session(id="oai-sess", system_prompt="you are a helper")

    async for _ in engine.run(session):
        pass

    assert session.state == RunState.DONE
    assert session.messages[-1].content == "all done"

    tool_msg = session.messages[1]
    assert tool_msg.role == Role.TOOL
    assert tool_msg.tool_results[0].call_id == "c1"
    assert tool_msg.tool_results[0].output == {"echo": {"x": 7}}

    first_request = client.completions.requests[0]
    assert first_request["messages"][0] == {
        "role": "system",
        "content": "you are a helper",
    }
    assert first_request["tools"][0]["type"] == "function"

    second_request = client.completions.requests[1]
    messages = second_request["messages"]
    assistant_msg = next(m for m in messages if m["role"] == "assistant")
    assert assistant_msg["tool_calls"][0]["id"] == "c1"
    tool_role_msg = next(m for m in messages if m["role"] == "tool")
    assert tool_role_msg["tool_call_id"] == "c1"
