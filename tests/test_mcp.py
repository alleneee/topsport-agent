from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from topsport_agent.engine.loop import Engine, EngineConfig
from topsport_agent.llm.provider import LLMResponse
from topsport_agent.llm.request import LLMRequest
from topsport_agent.mcp import (
    MCPClient,
    MCPManager,
    MCPToolSource,
    MCPTransport,
    build_mcp_meta_tools,
    load_mcp_config,
)
from topsport_agent.types.message import ToolCall
from topsport_agent.types.session import RunState, Session
from topsport_agent.types.tool import ToolContext, ToolSpec


@dataclass
class MockTool:
    name: str
    description: str = ""
    inputSchema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})


@dataclass
class MockTextContent:
    text: str
    type: str = "text"


@dataclass
class MockCallResult:
    content: list[MockTextContent]
    isError: bool = False
    structuredContent: dict[str, Any] | None = None


@dataclass
class MockListToolsResult:
    tools: list[MockTool]


@dataclass
class MockPrompt:
    name: str
    description: str = ""


@dataclass
class MockPromptMessage:
    role: str
    content: MockTextContent


@dataclass
class MockGetPromptResult:
    messages: list[MockPromptMessage]


@dataclass
class MockListPromptsResult:
    prompts: list[MockPrompt]


@dataclass
class MockResource:
    uri: str
    name: str = ""
    description: str = ""
    mimeType: str | None = None


@dataclass
class MockResourceContent:
    text: str


@dataclass
class MockReadResourceResult:
    contents: list[MockResourceContent]


@dataclass
class MockListResourcesResult:
    resources: list[MockResource]


class MockSession:
    def __init__(
        self,
        *,
        tools: list[MockTool] | None = None,
        prompts: list[MockPrompt] | None = None,
        resources: list[MockResource] | None = None,
    ) -> None:
        self.tools = tools or []
        self.prompts = prompts or []
        self.resources = resources or []
        self.call_log: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> MockListToolsResult:
        return MockListToolsResult(tools=self.tools)

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> MockCallResult:
        self.call_log.append((name, arguments))
        return MockCallResult(
            content=[MockTextContent(text=f"ran {name} with {arguments}")]
        )

    async def list_prompts(self) -> MockListPromptsResult:
        return MockListPromptsResult(prompts=self.prompts)

    async def get_prompt(
        self, name: str, arguments: dict[str, Any]
    ) -> MockGetPromptResult:
        return MockGetPromptResult(
            messages=[
                MockPromptMessage(
                    role="user",
                    content=MockTextContent(text=f"prompt {name} args={arguments}"),
                )
            ]
        )

    async def list_resources(self) -> MockListResourcesResult:
        return MockListResourcesResult(resources=self.resources)

    async def read_resource(self, uri: str) -> MockReadResourceResult:
        return MockReadResourceResult(
            contents=[MockResourceContent(text=f"content of {uri}")]
        )


def _factory_from(session: MockSession):
    @contextlib.asynccontextmanager
    async def factory():
        yield session

    return factory


@pytest.fixture
def cancel_event() -> asyncio.Event:
    return asyncio.Event()


def _ctx(cancel_event: asyncio.Event) -> ToolContext:
    return ToolContext(session_id="s1", call_id="c1", cancel_event=cancel_event)


def test_load_mcp_config_stdio_and_http(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["server.py"],
                        "env": {"DEBUG": "1"},
                    },
                    "remote": {
                        "transport": "http",
                        "url": "https://example.com/mcp",
                        "headers": {"Authorization": "Bearer xyz"},
                        "timeout": 60,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    configs = load_mcp_config(config_path)
    by_name = {c.name: c for c in configs}

    assert by_name["fs"].transport == MCPTransport.STDIO
    assert by_name["fs"].command == "python"
    assert by_name["fs"].args == ["server.py"]
    assert by_name["fs"].env == {"DEBUG": "1"}

    assert by_name["remote"].transport == MCPTransport.HTTP
    assert by_name["remote"].url == "https://example.com/mcp"
    assert by_name["remote"].headers == {"Authorization": "Bearer xyz"}
    assert by_name["remote"].timeout == 60.0


def test_load_mcp_config_stdio_missing_command_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"mcpServers": {"broken": {"transport": "stdio"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stdio requires 'command'"):
        load_mcp_config(path)


def test_load_mcp_config_http_missing_url_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"mcpServers": {"broken": {"transport": "http"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="http requires 'url'"):
        load_mcp_config(path)


def test_load_mcp_config_unknown_transport_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"mcpServers": {"broken": {"transport": "carrier-pigeon"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown transport"):
        load_mcp_config(path)


async def test_mcp_client_lazy_caches_tool_list():
    session = MockSession(tools=[MockTool(name="echo", description="Echo")])
    entries = 0

    @contextlib.asynccontextmanager
    async def factory():
        nonlocal entries
        entries += 1
        yield session

    client = MCPClient("test", factory)

    first = await client.list_tools()
    assert [t.name for t in first] == ["echo"]
    assert entries == 1

    second = await client.list_tools()
    assert [t.name for t in second] == ["echo"]
    assert entries == 1

    third = await client.list_tools(force_refresh=True)
    assert [t.name for t in third] == ["echo"]
    assert entries == 2


async def test_mcp_client_call_tool_opens_session_each_time():
    session = MockSession()
    entries = 0

    @contextlib.asynccontextmanager
    async def factory():
        nonlocal entries
        entries += 1
        yield session

    client = MCPClient("test", factory)
    await client.call_tool("echo", {"a": 1})
    await client.call_tool("echo", {"a": 2})

    assert entries == 2
    assert session.call_log == [("echo", {"a": 1}), ("echo", {"a": 2})]


async def test_mcp_client_invalidate_cache_forces_refetch():
    session = MockSession(tools=[MockTool(name="echo")])
    entries = 0

    @contextlib.asynccontextmanager
    async def factory():
        nonlocal entries
        entries += 1
        yield session

    client = MCPClient("test", factory)
    await client.list_tools()
    assert entries == 1

    client.invalidate_cache()
    await client.list_tools()
    assert entries == 2


async def test_mcp_tool_source_prefixes_tool_name(cancel_event: asyncio.Event):
    session = MockSession(tools=[MockTool(name="add", description="adds numbers")])
    client = MCPClient("math", _factory_from(session))
    source = MCPToolSource(client)

    specs = await source.list_tools()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "math.add"
    assert spec.description == "adds numbers"

    result = await spec.handler({"a": 3}, _ctx(cancel_event))
    assert result["is_error"] is False
    assert "ran add" in result["text"]
    assert result["structured"] is None


async def test_mcp_tool_source_swallows_list_tools_exception():
    class BrokenSession:
        async def list_tools(self):
            raise RuntimeError("connection lost")

    @contextlib.asynccontextmanager
    async def factory():
        yield BrokenSession()

    client = MCPClient("broken", factory)
    source = MCPToolSource(client)
    specs = await source.list_tools()
    assert specs == []


async def test_mcp_tool_handler_reports_tool_exception(cancel_event: asyncio.Event):
    class ErroringSession:
        async def list_tools(self):
            return MockListToolsResult(tools=[MockTool(name="bad")])

        async def call_tool(self, name, arguments):
            raise RuntimeError("tool broke")

    @contextlib.asynccontextmanager
    async def factory():
        yield ErroringSession()

    client = MCPClient("err", factory)
    source = MCPToolSource(client)
    specs = await source.list_tools()

    result = await specs[0].handler({}, _ctx(cancel_event))
    assert result["is_error"] is True
    assert "tool broke" in result["error"]


async def test_mcp_manager_registers_and_exposes_clients():
    session = MockSession(tools=[MockTool(name="a")])
    client = MCPClient("srv1", _factory_from(session))
    manager = MCPManager()
    manager.register(client)

    assert manager.get("srv1") is client
    assert manager.get("missing") is None
    assert len(manager.clients()) == 1
    assert len(manager.tool_sources()) == 1


async def test_mcp_meta_tools_list_and_get_prompt(cancel_event: asyncio.Event):
    session = MockSession(
        prompts=[MockPrompt(name="greet", description="say hi")]
    )
    client = MCPClient("p1", _factory_from(session))
    manager = MCPManager()
    manager.register(client)

    tools = {t.name: t for t in build_mcp_meta_tools(manager)}

    list_result = await tools["list_mcp_prompts"].handler({}, _ctx(cancel_event))
    servers = list_result["servers"]
    assert len(servers) == 1
    assert servers[0]["server"] == "p1"
    assert servers[0]["prompts"] == [{"name": "greet", "description": "say hi"}]

    get_result = await tools["get_mcp_prompt"].handler(
        {"server": "p1", "name": "greet", "arguments": {"who": "world"}},
        _ctx(cancel_event),
    )
    assert get_result["server"] == "p1"
    assert get_result["name"] == "greet"
    assert get_result["messages"][0]["role"] == "user"
    assert "world" in get_result["messages"][0]["text"]


async def test_mcp_meta_tools_list_and_read_resource(cancel_event: asyncio.Event):
    session = MockSession(
        resources=[
            MockResource(
                uri="file:///tmp/test.md",
                name="test",
                mimeType="text/markdown",
            )
        ]
    )
    client = MCPClient("r1", _factory_from(session))
    manager = MCPManager()
    manager.register(client)

    tools = {t.name: t for t in build_mcp_meta_tools(manager)}

    list_result = await tools["list_mcp_resources"].handler(
        {"server": "r1"}, _ctx(cancel_event)
    )
    assert list_result["server"] == "r1"
    assert list_result["resources"][0]["uri"] == "file:///tmp/test.md"
    assert list_result["resources"][0]["mimeType"] == "text/markdown"

    read_result = await tools["read_mcp_resource"].handler(
        {"server": "r1", "uri": "file:///tmp/test.md"},
        _ctx(cancel_event),
    )
    assert "content of file:///tmp/test.md" in read_result["text"]


async def test_mcp_meta_tools_unknown_server(cancel_event: asyncio.Event):
    manager = MCPManager()
    tools = {t.name: t for t in build_mcp_meta_tools(manager)}

    result = await tools["get_mcp_prompt"].handler(
        {"server": "nope", "name": "x"}, _ctx(cancel_event)
    )
    assert "not found" in result["error"]


class _ScriptedProvider:
    name = "scripted"

    def __init__(self, turns: list[LLMResponse]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.seen_tools: list[list[ToolSpec]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.seen_tools.append(list(request.tools))
        if self._index >= len(self._turns):
            return LLMResponse(text="fallback", finish_reason="stop")
        turn = self._turns[self._index]
        self._index += 1
        return turn


async def test_mcp_tool_source_integrates_with_engine():
    session = MockSession(tools=[MockTool(name="echo", description="echo")])
    client = MCPClient("mcp1", _factory_from(session))
    source = MCPToolSource(client)

    provider = _ScriptedProvider(
        [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c1", name="mcp1.echo", arguments={"x": 1})
                ],
                finish_reason="tool_use",
            ),
            LLMResponse(text="done", finish_reason="stop"),
        ]
    )
    engine = Engine(
        provider,
        tools=[],
        config=EngineConfig(model="fake"),
        tool_sources=[source],
    )
    sess = Session(id="mcp-sess", system_prompt="sys")

    async for _ in engine.run(sess):
        pass

    assert sess.state == RunState.DONE
    assert any(t.name == "mcp1.echo" for t in provider.seen_tools[0])

    tool_result_msg = sess.messages[1]
    assert tool_result_msg.tool_results[0].is_error is False
    output = tool_result_msg.tool_results[0].output
    assert output["is_error"] is False
    assert "ran echo" in output["text"]
