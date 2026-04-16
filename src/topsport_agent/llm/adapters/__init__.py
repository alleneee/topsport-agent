from .anthropic import AnthropicMessagesAdapter
from .openai_chat import OpenAIChatAdapter

__all__ = [
    "AnthropicMessagesAdapter",
    "AnthropicProvider",
    "OpenAIChatAdapter",
    "OpenAIChatProvider",
]


def __getattr__(name: str):
    if name == "AnthropicProvider":
        from ..providers import AnthropicProvider

        return AnthropicProvider
    if name == "OpenAIChatProvider":
        from ..providers import OpenAIChatProvider

        return OpenAIChatProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
