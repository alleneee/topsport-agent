from topsport_agent.llm.response import (
    LLM_RESPONSE_EXTRA_KEY,
    ProviderResponseMetadata,
    get_response_metadata,
    wrap_response_metadata,
)


def test_wrap_response_metadata_uses_stable_namespace():
    wrapped = wrap_response_metadata(
        ProviderResponseMetadata(
            provider="openai",
            assistant_blocks=[{"type": "text", "text": "ok"}],
        )
    )

    assert wrapped == {
        LLM_RESPONSE_EXTRA_KEY: {
            "provider": "openai",
            "assistant_blocks": [{"type": "text", "text": "ok"}],
        }
    }


def test_get_response_metadata_reads_from_namespace():
    extra = {
        "other": {"keep": True},
        LLM_RESPONSE_EXTRA_KEY: {
            "provider": "anthropic",
            "assistant_blocks": [{"type": "tool_use", "id": "c1", "name": "echo", "input": {"x": 1}}],
        },
    }

    assert get_response_metadata(extra) == ProviderResponseMetadata(
        provider="anthropic",
        assistant_blocks=[{"type": "tool_use", "id": "c1", "name": "echo", "input": {"x": 1}}],
    )
