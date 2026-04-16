from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.adapters import AnthropicProvider
from topsport_agent.llm.request import LLMRequest
from topsport_agent.llm.response import ProviderResponseMetadata
from topsport_agent.types.message import Message, Role, ToolCall, ToolResult
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


@dataclass
class MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class MockToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class MockUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MockMessageResponse:
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: MockUsage = field(default_factory=MockUsage)


class MockMessages:
    def __init__(self, responses: list[MockMessageResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> MockMessageResponse:
        self.requests.append(kwargs)
        if self._index >= len(self._responses):
            return MockMessageResponse(
                content=[MockTextBlock(text="fallback")],
                stop_reason="end_turn",
            )
        response = self._responses[self._index]
        self._index += 1
        return response


class MockAnthropicClient:
    def __init__(self, responses: list[MockMessageResponse]) -> None:
        self.messages = MockMessages(responses)

    async def create(self, payload: dict[str, Any]) -> MockMessageResponse:
        return await self.messages.create(**payload)


def _text_response(text: str, *, stop_reason: str = "end_turn") -> MockMessageResponse:
    return MockMessageResponse(
        content=[MockTextBlock(text=text)],
        stop_reason=stop_reason,
        usage=MockUsage(input_tokens=10, output_tokens=5),
    )


def _tool_use_response(
    call_id: str, name: str, arguments: dict[str, Any]
) -> MockMessageResponse:
    return MockMessageResponse(
        content=[MockToolUseBlock(id=call_id, name=name, input=arguments)],
        stop_reason="tool_use",
        usage=MockUsage(input_tokens=12, output_tokens=8),
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


async def test_converts_simple_user_message_and_system_prompt():
    client = MockAnthropicClient([_text_response("hi there")])
    provider = AnthropicProvider(client=client)

    messages = [Message(role=Role.USER, content="hello")]
    response = await provider.complete(
        LLMRequest(model="claude-sonnet-4-5", messages=messages)
    )

    assert client.messages.requests[0]["model"] == "claude-sonnet-4-5"
    assert client.messages.requests[0]["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert "system" not in client.messages.requests[0]
    assert response.text == "hi there"
    assert response.finish_reason == "end_turn"
    assert response.usage == {"input_tokens": 10, "output_tokens": 5}


async def test_system_messages_lift_to_system_param():
    client = MockAnthropicClient([_text_response("ok")])
    provider = AnthropicProvider(client=client)

    messages = [
        Message(role=Role.SYSTEM, content="you are a helpful bot"),
        Message(role=Role.SYSTEM, content="respond in english"),
        Message(role=Role.USER, content="hi"),
    ]
    await provider.complete(LLMRequest(model="m", messages=messages))

    request = client.messages.requests[0]
    assert request["system"] == "you are a helpful bot\n\nrespond in english"
    assert all(m["role"] != "system" for m in request["messages"])


async def test_assistant_tool_calls_render_as_content_blocks():
    client = MockAnthropicClient([_text_response("ok")])
    provider = AnthropicProvider(client=client)

    messages = [
        Message(role=Role.USER, content="use echo"),
        Message(
            role=Role.ASSISTANT,
            content="let me try",
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": 1})],
        ),
        Message(
            role=Role.TOOL,
            tool_results=[ToolResult(call_id="c1", output={"result": 1})],
        ),
        Message(role=Role.USER, content="continue"),
    ]
    await provider.complete(LLMRequest(model="m", messages=messages))

    request_msgs = client.messages.requests[0]["messages"]
    assert request_msgs[0] == {"role": "user", "content": "use echo"}
    assistant = request_msgs[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "let me try"}
    assert assistant["content"][1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "echo",
        "input": {"x": 1},
    }

    tool_result_msg = request_msgs[2]
    assert tool_result_msg["role"] == "user"
    assert len(tool_result_msg["content"]) == 1
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "c1"

    assert request_msgs[3] == {"role": "user", "content": "continue"}


async def test_multiple_consecutive_tool_results_merge_into_single_user_message():
    client = MockAnthropicClient([_text_response("done")])
    provider = AnthropicProvider(client=client)

    messages = [
        Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[
                ToolCall(id="c1", name="echo", arguments={}),
                ToolCall(id="c2", name="echo", arguments={}),
            ],
        ),
        Message(
            role=Role.TOOL,
            tool_results=[ToolResult(call_id="c1", output="a")],
        ),
        Message(
            role=Role.TOOL,
            tool_results=[ToolResult(call_id="c2", output="b")],
        ),
    ]
    await provider.complete(LLMRequest(model="m", messages=messages))

    request_msgs = client.messages.requests[0]["messages"]
    assert len(request_msgs) == 2
    assistant = request_msgs[0]
    assert assistant["role"] == "assistant"
    assert len(assistant["content"]) == 2

    tool_results = request_msgs[1]
    assert tool_results["role"] == "user"
    assert len(tool_results["content"]) == 2
    assert tool_results["content"][0]["tool_use_id"] == "c1"
    assert tool_results["content"][1]["tool_use_id"] == "c2"


async def test_error_tool_result_is_marked():
    client = MockAnthropicClient([_text_response("ok")])
    provider = AnthropicProvider(client=client)

    messages = [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={})],
        ),
        Message(
            role=Role.TOOL,
            tool_results=[
                ToolResult(call_id="c1", output="boom", is_error=True)
            ],
        ),
    ]
    await provider.complete(LLMRequest(model="m", messages=messages))

    tool_result = client.messages.requests[0]["messages"][1]["content"][0]
    assert tool_result["is_error"] is True


async def test_dict_tool_output_is_json_encoded():
    client = MockAnthropicClient([_text_response("ok")])
    provider = AnthropicProvider(client=client)

    messages = [
        Message(
            role=Role.ASSISTANT,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={})],
        ),
        Message(
            role=Role.TOOL,
            tool_results=[
                ToolResult(call_id="c1", output={"nested": {"value": 42}})
            ],
        ),
    ]
    await provider.complete(LLMRequest(model="m", messages=messages))

    tool_result = client.messages.requests[0]["messages"][1]["content"][0]
    text_block = tool_result["content"][0]
    assert text_block["type"] == "text"
    assert '"value": 42' in text_block["text"]


async def test_tool_specs_convert_to_anthropic_format():
    client = MockAnthropicClient([_text_response("ok")])
    provider = AnthropicProvider(client=client)

    await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="hi")], tools=[_echo_tool()])
    )

    tools = client.messages.requests[0]["tools"]
    assert tools == [
        {
            "name": "echo",
            "description": "echo back arguments",
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        }
    ]


async def test_response_tool_use_blocks_become_tool_calls():
    client = MockAnthropicClient(
        [_tool_use_response("c1", "echo", {"x": 5})]
    )
    provider = AnthropicProvider(client=client)

    response = await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="run echo")], tools=[_echo_tool()])
    )

    assert response.text is None
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "c1"
    assert call.name == "echo"
    assert call.arguments == {"x": 5}
    assert response.finish_reason == "tool_use"
    assert response.usage == {"input_tokens": 12, "output_tokens": 8}
    assert response.response_metadata == ProviderResponseMetadata(
        provider="anthropic",
        assistant_blocks=[
            {
                "type": "tool_use",
                "id": "c1",
                "name": "echo",
                "input": {"x": 5},
            }
        ],
    )


async def test_response_mixes_text_and_tool_use():
    response_mock = MockMessageResponse(
        content=[
            MockTextBlock(text="thinking about it"),
            MockToolUseBlock(id="c1", name="echo", input={"x": 1}),
        ],
        stop_reason="tool_use",
    )
    client = MockAnthropicClient([response_mock])
    provider = AnthropicProvider(client=client)

    response = await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="hi")], tools=[_echo_tool()])
    )

    assert response.text == "thinking about it"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "c1"


async def test_thinking_from_constructor_is_forwarded():
    client = MockAnthropicClient([_text_response("ok")])
    provider = AnthropicProvider(client=client, thinking_budget=2048)

    await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="hi")])
    )

    request = client.messages.requests[0]
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 2048}


async def test_thinking_from_extra_overrides_constructor():
    client = MockAnthropicClient([_text_response("ok")])
    provider = AnthropicProvider(client=client, thinking_budget=1024)

    await provider.complete(
        LLMRequest(
            model="m",
            messages=[Message(role=Role.USER, content="hi")],
            provider_options={"anthropic": {"thinking": {"type": "enabled", "budget_tokens": 8192}}},
        )
    )

    request = client.messages.requests[0]
    assert request["thinking"]["budget_tokens"] == 8192


async def test_extra_max_tokens_honored():
    client = MockAnthropicClient([_text_response("ok")])
    provider = AnthropicProvider(client=client, max_tokens=100)

    await provider.complete(
        LLMRequest(
            model="m",
            messages=[Message(role=Role.USER, content="hi")],
            max_output_tokens=2048,
        )
    )

    assert client.messages.requests[0]["max_tokens"] == 2048


async def test_thinking_block_in_response_is_ignored_for_text():
    @dataclass
    class MockThinkingBlock:
        thinking: str
        type: str = "thinking"

    response_mock = MockMessageResponse(
        content=[
            MockThinkingBlock(thinking="let me think..."),
            MockTextBlock(text="answer: 42"),
        ],
        stop_reason="end_turn",
    )
    client = MockAnthropicClient([response_mock])
    provider = AnthropicProvider(client=client)

    response = await provider.complete(
        LLMRequest(model="m", messages=[Message(role=Role.USER, content="hi")])
    )

    assert response.text == "answer: 42"
    assert response.tool_calls == []


async def test_engine_integration_with_anthropic_adapter():
    tool_use_resp = MockMessageResponse(
        content=[MockToolUseBlock(id="c1", name="echo", input={"x": 7})],
        stop_reason="tool_use",
        usage=MockUsage(input_tokens=15, output_tokens=10),
    )
    final_resp = MockMessageResponse(
        content=[MockTextBlock(text="all done")],
        stop_reason="end_turn",
        usage=MockUsage(input_tokens=20, output_tokens=12),
    )
    client = MockAnthropicClient([tool_use_resp, final_resp])
    provider = AnthropicProvider(client=client)

    engine = Engine(
        provider,
        tools=[_echo_tool()],
        config=EngineConfig(model="claude-sonnet-4-5"),
    )
    session = Session(id="anth-sess", system_prompt="you are a helper")

    async for _ in engine.run(session):
        pass

    assert session.state == RunState.DONE
    assert session.messages[-1].content == "all done"
    assert session.messages[-1].role == Role.ASSISTANT

    tool_msg = session.messages[1]
    assert tool_msg.role == Role.TOOL
    assert tool_msg.tool_results[0].call_id == "c1"
    assert tool_msg.tool_results[0].output == {"echo": {"x": 7}}

    first_request = client.messages.requests[0]
    assert first_request["system"] == "you are a helper"
    assert first_request["messages"] == []
    assert first_request["tools"][0]["name"] == "echo"

    second_request = client.messages.requests[1]
    assert second_request["messages"][0]["role"] == "assistant"
    assert second_request["messages"][0]["content"][0]["type"] == "tool_use"
    assert second_request["messages"][1]["role"] == "user"
    assert (
        second_request["messages"][1]["content"][0]["type"] == "tool_result"
    )
