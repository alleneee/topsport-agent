"""Agent 级 E2E：LLM 通过工具调用驱动 OpenSandbox 执行实际工作。

验证目标：
1. Agent 能发现并正确调用 sandbox_write_file / sandbox_shell（schema 正确）
2. 同一 session 的多个 tool call 命中同一个 sandbox（session 绑定正确）
3. 任务完成后 pool.close_all() + agent.close() 清理干净（生命周期正确）

前置条件：
- `docker compose -f /tmp/OpenSandbox/server/docker-compose.example.yaml up -d opensandbox-server`
- 本项目 .env 里有 API_KEY / BASE_URL / MODEL（anthropic 或 openai 格式）

用法：`uv run python scripts/sandbox_agent_e2e.py`
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
from datetime import timedelta
from pathlib import Path

from topsport_agent.agent.base import Agent, AgentConfig
from topsport_agent.sandbox import OpenSandboxPool, OpenSandboxToolSource
from topsport_agent.types.events import EventType


def _load_dotenv(path: Path) -> None:
    """最小 .env 加载：KEY=VALUE，忽略注释与空行。
    不做权限校验（server main.py 才严格）；探索脚本够用。
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_model(model_str: str) -> tuple[str, str]:
    provider, _, model = model_str.partition("/")
    return provider.strip().lower(), model.strip()


def _make_provider(provider_name: str, api_key: str, base_url: str | None):
    if provider_name == "anthropic":
        mod = importlib.import_module("topsport_agent.llm.providers.anthropic")
        return mod.AnthropicProvider(api_key=api_key, base_url=base_url, max_tokens=2048)
    mod = importlib.import_module("topsport_agent.llm.providers.openai_chat")
    return mod.OpenAIChatProvider(api_key=api_key, base_url=base_url, max_tokens=2048)


SYSTEM_PROMPT = """You are a minimal test agent with access to a sandbox environment.

Available tools (all paths must be absolute):
- sandbox_shell(command): run a shell command
- sandbox_read_file(path): read a text file
- sandbox_write_file(path, content): write a file

Use them to complete the user's task. When done, reply with a short summary.
Do NOT use any other tool even if you think of one."""


USER_TASK = (
    "Please do exactly these two steps in the sandbox:\n"
    "1. Write the text 'hello topsport-agent' to /tmp/greet.txt\n"
    "2. Use the shell to `cat /tmp/greet.txt` and show me the output\n"
    "Then summarize what you did."
)


async def main() -> int:
    _load_dotenv(Path(".env"))
    api_key = os.environ.get("API_KEY", "").strip()
    base_url = os.environ.get("BASE_URL") or None
    model_str = os.environ.get("MODEL", "").strip()
    if not api_key or not model_str:
        print("need API_KEY and MODEL in env / .env", file=sys.stderr)
        return 2

    provider_name, model = _parse_model(model_str)
    provider = _make_provider(provider_name, api_key, base_url)

    pool = OpenSandboxPool.from_config(
        domain="localhost:8090",
        image="ubuntu",
        protocol="http",
        use_server_proxy=True,
        request_timeout=timedelta(seconds=120),
    )
    tool_source = OpenSandboxToolSource(pool)

    # 关掉 skills/memory/plugins/browser/file_ops，让 LLM 只看到 3 个 sandbox_* 工具
    config = AgentConfig(
        name="sandbox-e2e",
        description="sandbox e2e agent",
        system_prompt=SYSTEM_PROMPT,
        model=model,
        max_steps=8,
        enable_skills=False,
        enable_memory=False,
        enable_plugins=False,
        enable_browser=False,
        stream=False,
        extra_tool_sources=[tool_source],
    )
    agent = Agent.from_config(provider, config)
    session = agent.new_session()

    print(f"=== provider={provider_name} model={model} ===")
    print(f"=== user task ===\n{USER_TASK}\n=== run ===")

    tool_calls: list[tuple[str, bool]] = []
    final_text_parts: list[str] = []
    error_events: list[str] = []

    try:
        async for event in agent.run(USER_TASK, session):
            payload = event.payload or {}
            if event.type == EventType.TOOL_CALL_START:
                name = payload.get("name", "?")
                args = payload.get("arguments") or payload.get("args") or {}
                print(f"  [tool start] {name}  args={args}")
            elif event.type == EventType.TOOL_CALL_END:
                name = payload.get("name", "?")
                is_err = bool(payload.get("is_error"))
                tool_calls.append((name, is_err))
                print(f"  [tool end]   {name}  error={is_err}")
            elif event.type == EventType.LLM_CALL_END:
                usage = payload.get("usage") or {}
                print(f"  [llm end]    tokens: in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')}")
            elif event.type == EventType.ERROR:
                msg = f"{payload.get('kind')}: {payload.get('message')}"
                error_events.append(msg)
                print(f"  [ERROR] {msg}")

        # 取最终 assistant 文本
        for msg in reversed(session.messages):
            role = getattr(msg.role, "value", msg.role)
            if str(role) in ("assistant",) and msg.content:
                final_text_parts.append(str(msg.content))
                break
    finally:
        await agent.close()
        await pool.close_all()

    print("\n=== result ===")
    print("tool_calls:", tool_calls)
    print("errors:", error_events)
    print("final text:", final_text_parts[0] if final_text_parts else "(none)")

    # 断言
    names_called = {name for name, _ in tool_calls}
    expected = {"sandbox_write_file", "sandbox_shell"}
    missing = expected - names_called
    if missing:
        print(f"\nFAIL: missing tool calls: {missing}")
        return 1
    if any(err for _, err in tool_calls):
        print("\nFAIL: some tool call reported is_error=True")
        return 1
    if error_events:
        print("\nFAIL: ERROR events captured")
        return 1
    print("\nPASS: agent drove sandbox via write + shell tools, clean lifecycle.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
