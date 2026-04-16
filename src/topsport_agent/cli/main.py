"""topsport-agent CLI: 交互式 REPL，验证引擎核心链路。

用法:
    uv run topsport-agent --model anthropic/claude-sonnet-4-5
    uv run topsport-agent --model openai/gpt-4o

MODEL 格式为 provider/model-name，provider 支持 anthropic 和 openai。
API_KEY 和 BASE_URL 从 .env 或环境变量读取，不带厂商前缀。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
from pathlib import Path
import sys
import uuid

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..engine import Engine, EngineConfig
from ..types.events import EventType
from ..types.message import Message, Role
from ..types.session import RunState, Session
from .tools import builtin_tools

console = Console()

_prompt_style = Style.from_dict({
    "prompt": "bold cyan",
})


def _parse_model(model_str: str) -> tuple[str, str]:
    """解析 'provider/model-name' 格式，返回 (provider, model)。"""
    if "/" not in model_str:
        console.print(f"[red]MODEL must be 'provider/model-name', got: {model_str}[/]")
        console.print("  examples: anthropic/claude-sonnet-4-5, openai/gpt-4o")
        sys.exit(1)
    provider, _, model = model_str.partition("/")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in ("anthropic", "openai"):
        console.print(f"[red]unknown provider: {provider}, use 'anthropic' or 'openai'[/]")
        sys.exit(1)
    if not model:
        console.print("[red]model name is empty[/]")
        sys.exit(1)
    return provider, model


def _make_provider(provider_name: str, api_key: str, base_url: str | None) -> object:
    """根据 provider 名称动态加载并构造 Provider，直接传入 api_key/base_url。"""
    if provider_name == "anthropic":
        mod_name = "topsport_agent.llm.providers.anthropic"
        mod = importlib.import_module(mod_name)
        return mod.AnthropicProvider(
            api_key=api_key,
            base_url=base_url,
            max_tokens=4096,
        )
    else:
        mod_name = "topsport_agent.llm.providers.openai_chat"
        mod = importlib.import_module(mod_name)
        return mod.OpenAIChatProvider(
            api_key=api_key,
            base_url=base_url,
            max_tokens=4096,
        )


def _print_event(event_type: EventType, payload: dict) -> None:
    """打印关键事件，非关键事件跳过。"""
    if event_type == EventType.TOOL_CALL_START:
        name = payload.get("name", "?")
        console.print(f"  [dim]tool calling:[/] [yellow]{name}[/]")
    elif event_type == EventType.TOOL_CALL_END:
        name = payload.get("name", "?")
        if payload.get("is_error"):
            console.print(f"  [dim]tool done:[/] [red]{name} (error)[/]")
        else:
            console.print(f"  [dim]tool done:[/] [green]{name}[/]")
    elif event_type == EventType.LLM_CALL_END:
        usage = payload.get("usage", {})
        inp = usage.get("input_tokens", "?")
        out = usage.get("output_tokens", "?")
        console.print(f"  [dim]tokens: {inp} in / {out} out[/]")
    elif event_type == EventType.ERROR:
        kind = payload.get("kind", "?")
        msg = payload.get("message", "")
        console.print(f"  [bold red]error:[/] {kind}: {msg}")


def _try_make_browser() -> tuple[object | None, list[object]]:
    """尝试创建浏览器工具源。Playwright 未安装时静默跳过。"""
    try:
        # 先验证 playwright 可导入，避免注册了工具但调用全部失败
        pw_name = "playwright.async_api"
        importlib.import_module(pw_name)
    except ImportError:
        return None, []
    try:
        from ..browser import BrowserClient, BrowserConfig, BrowserToolSource
        client = BrowserClient.from_config(BrowserConfig(headless=True))
        source = BrowserToolSource(client)
        return client, [source]
    except Exception:
        return None, []


async def _run_loop(provider: object, model: str, system_prompt: str) -> None:
    """主交互循环：prompt_toolkit 读取输入 -> 引擎推理 -> rich 输出结果。"""
    browser_client, browser_sources = _try_make_browser()

    engine = Engine(
        provider=provider,  # type: ignore[arg-type]
        tools=builtin_tools(),
        config=EngineConfig(model=model, max_steps=10),
        tool_sources=browser_sources,  # type: ignore[arg-type]
    )

    session = Session(
        id=str(uuid.uuid4()),
        system_prompt=system_prompt,
    )

    # 历史记录持久化到用户目录
    history_dir = Path.home() / ".topsport-agent"
    history_dir.mkdir(exist_ok=True)
    history_file = history_dir / "history.txt"

    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
    )

    all_tool_names = [t.name for t in builtin_tools()]
    if browser_sources:
        all_tool_names.extend(["browser_navigate", "browser_snapshot", "browser_click",
                               "browser_type", "browser_screenshot", "browser_get_text"])
    tools_text = ", ".join(all_tool_names)
    console.print(Panel.fit(
        f"[bold]model:[/] {model}\n[bold]tools:[/] {tools_text}\n[bold]session:[/] {session.id[:8]}...",
        title="[bold cyan]topsport-agent[/]",
        border_style="cyan",
    ))
    console.print("[dim]type 'quit' or Ctrl+D to exit[/]\n")

    while True:
        try:
            raw = await prompt_session.prompt_async(
                [("class:prompt", "you> ")],
                style=_prompt_style,
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            break

        user_input = raw.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]bye[/]")
            break

        session.messages.append(Message(role=Role.USER, content=user_input))
        session.state = RunState.IDLE

        assistant_text: str | None = None

        async for event in engine.run(session):
            _print_event(event.type, event.payload)

            if event.type == EventType.MESSAGE_APPENDED:
                if event.payload.get("role") == "assistant":
                    last_msg = session.messages[-1] if session.messages else None
                    if last_msg and last_msg.role == Role.ASSISTANT and last_msg.content:
                        assistant_text = last_msg.content

        if assistant_text:
            console.print()
            console.print(Markdown(assistant_text))
            console.print()
        else:
            console.print()

        engine.reset_cancel()

    if browser_client is not None:
        await browser_client.close()  # type: ignore[union-attr]


def _load_dotenv() -> None:
    """从项目根目录加载 .env 文件，已有的环境变量不覆盖。"""
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return
    with open(env_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        prog="topsport-agent",
        description="topsport-agent interactive CLI",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="provider/model (e.g. anthropic/claude-sonnet-4-5, openai/gpt-4o)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="system prompt",
    )

    args = parser.parse_args()

    model_str = args.model or os.environ.get("MODEL")
    if not model_str:
        console.print("[red]--model or MODEL env var is required[/]")
        console.print("  format: provider/model-name")
        console.print("  examples: anthropic/claude-sonnet-4-5, openai/gpt-4o")
        sys.exit(1)

    provider_name, model = _parse_model(model_str)

    api_key = os.environ.get("API_KEY", "")
    base_url = os.environ.get("BASE_URL")

    if not api_key:
        console.print("[red]API_KEY not set[/]")
        sys.exit(1)

    try:
        provider = _make_provider(provider_name, api_key, base_url)
    except ImportError as exc:
        console.print(f"[red]cannot import {provider_name} SDK: {exc}[/]")
        console.print("  run: uv sync --group llm")
        sys.exit(1)

    system_prompt = args.system
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant with browser control capabilities. "
            "You can navigate web pages, click elements, type text, take screenshots, and extract text. "
            "When asked to visit a website, use browser_navigate first, then use the @refs from the snapshot "
            "to interact with page elements (browser_click, browser_type). "
            "Use browser_get_text to extract page content. "
            "Use browser_screenshot to capture visual evidence. "
            "You also have: echo, current_time, calc tools."
        )

    asyncio.run(_run_loop(provider, model, system_prompt))


if __name__ == "__main__":
    main()
