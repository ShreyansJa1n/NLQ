from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nl_db.config import load_settings
from nl_db.llm.anthropic_provider import AnthropicProvider
from nl_db.llm.openai_compatible import OpenAICompatibleProvider
from nl_db.llm.openai_provider import OpenAIProvider
from nl_db.llm.provider import Message, ToolDef, ToolsNotSupportedError
from nl_db.llm.registry import build_provider


class FakeAnthropicClient:
    """Fake anthropic client.

    Default returns a single text block. Inject `next_response` to swap in a
    pre-built SimpleNamespace (e.g. with a tool_use block) for a single call.
    """

    def __init__(self) -> None:
        self.messages = self
        self.last_call: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []
        self.next_response: Any | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_call = kwargs
        self.calls.append(kwargs)
        if self.next_response is not None:
            r = self.next_response
            self.next_response = None
            return r
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="SELECT 1;")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            stop_reason="end_turn",
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.last_call: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_call = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="SELECT 2;"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
        )


def test_anthropic_provider_separates_system_from_convo() -> None:
    client = FakeAnthropicClient()
    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="x", client=client)
    result = p.chat(
        [
            Message(role="system", content="you are a sql expert"),
            Message(role="user", content="hi"),
        ]
    )
    assert result.text == "SELECT 1;"
    assert result.input_tokens == 12
    assert result.output_tokens == 4
    assert client.last_call is not None
    assert client.last_call["system"] == "you are a sql expert"
    assert client.last_call["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_provider_forwards_messages() -> None:
    client = FakeOpenAIClient()
    p = OpenAIProvider(model="gpt-4o", api_key="x", client=client)
    result = p.chat(
        [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ]
    )
    assert result.text == "SELECT 2;"
    assert result.input_tokens == 10
    assert client.last_call is not None
    assert client.last_call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_openai_compatible_provider_uses_base_url() -> None:
    client = FakeOpenAIClient()
    p = OpenAICompatibleProvider(
        model="local-llama",
        base_url="http://localhost:8080/v1",
        api_key=None,
        client=client,
    )
    result = p.chat([Message(role="user", content="hi")])
    assert result.text == "SELECT 2;"


def test_registry_builds_anthropic(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    settings = load_settings(config_path=tmp_path / "missing.toml")
    provider = build_provider(settings)
    assert provider.name == "anthropic"
    assert isinstance(provider, AnthropicProvider)


def test_registry_builds_openai_compatible(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("NL_DB_PROVIDER__NAME", "openai_compatible")
    monkeypatch.setenv("NL_DB_PROVIDER__BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("NL_DB_PROVIDER__MODEL", "local-llama")
    settings = load_settings(config_path=tmp_path / "missing.toml")
    provider = build_provider(settings)
    assert provider.name == "openai_compatible"
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "local-llama"


# Tool-calling capability declaration ----------------------------------------

def test_supports_tools_capability_per_provider() -> None:
    # Static metadata — no construction-time probes.
    assert AnthropicProvider.supports_tools is True
    assert OpenAIProvider.supports_tools is True
    assert OpenAICompatibleProvider.supports_tools is None


def test_anthropic_passes_tools_in_wire_format() -> None:
    client = FakeAnthropicClient()
    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="x", client=client)
    tool = ToolDef(
        name="list_tables",
        description="list table names",
        input_schema={"type": "object", "properties": {}},
    )
    p.chat([Message(role="user", content="hi")], tools=(tool,))
    assert client.last_call is not None
    assert client.last_call["tools"] == [
        {
            "name": "list_tables",
            "description": "list table names",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_anthropic_parses_tool_use_blocks() -> None:
    client = FakeAnthropicClient()
    client.next_response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tool_abc",
                name="describe_table",
                input={"table_name": "users"},
            ),
        ],
        usage=SimpleNamespace(input_tokens=20, output_tokens=8),
        stop_reason="tool_use",
    )
    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="x", client=client)
    tool = ToolDef(name="describe_table", description="", input_schema={})
    result = p.chat([Message(role="user", content="hi")], tools=(tool,))

    assert result.text == ""
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "tool_abc"
    assert call.name == "describe_table"
    assert call.arguments == {"table_name": "users"}
    assert result.stop_reason == "tool_use"


def test_anthropic_handles_mixed_text_and_tool_use() -> None:
    client = FakeAnthropicClient()
    client.next_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Let me check the schema first."),
            SimpleNamespace(
                type="tool_use",
                id="tool_xyz",
                name="list_tables",
                input={},
            ),
        ],
        usage=SimpleNamespace(input_tokens=15, output_tokens=10),
        stop_reason="tool_use",
    )
    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="x", client=client)
    tool = ToolDef(name="list_tables", description="", input_schema={})
    result = p.chat([Message(role="user", content="hi")], tools=(tool,))
    assert "Let me check" in result.text
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "list_tables"


def test_anthropic_translates_tool_messages_to_tool_result_blocks() -> None:
    """A loop-back: assistant requested a tool, we feed the result back as a
    role='tool' Message. Anthropic expects this packaged as a user message
    with a tool_result content block referencing the tool_use_id."""
    client = FakeAnthropicClient()
    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="x", client=client)
    p.chat(
        [
            Message(role="user", content="what tables are there?"),
            Message(role="assistant", content=""),  # the tool_use message
            Message(
                role="tool",
                content='{"tables": ["users", "orders"]}',
                tool_call_id="tool_abc",
                tool_name="list_tables",
            ),
        ]
    )
    assert client.last_call is not None
    convo = client.last_call["messages"]
    # The tool message becomes a user message with a tool_result content block.
    tool_result_msg = convo[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "tool_abc",
            "content": '{"tables": ["users", "orders"]}',
        }
    ]


def test_chat_with_tools_raises_until_wired_openai() -> None:
    client = FakeOpenAIClient()
    p = OpenAIProvider(model="gpt-4o", api_key="x", client=client)
    dummy_tool = ToolDef(
        name="list_tables",
        description="list table names",
        input_schema={"type": "object", "properties": {}},
    )
    with pytest.raises(ToolsNotSupportedError):
        p.chat([Message(role="user", content="hi")], tools=(dummy_tool,))


def test_chat_without_tools_unaffected_anthropic() -> None:
    # Same call as the existing test, just confirms the new tools= kwarg
    # doesn't break the no-tools path.
    client = FakeAnthropicClient()
    p = AnthropicProvider(model="claude-sonnet-4-6", api_key="x", client=client)
    result = p.chat([Message(role="user", content="hi")])
    assert result.text == "SELECT 1;"
    assert result.tool_calls == ()


def test_message_with_tool_role_constructs() -> None:
    m = Message(
        role="tool",
        content='{"tables": ["users", "posts"]}',
        tool_call_id="call_abc",
        tool_name="list_tables",
    )
    assert m.role == "tool"
    assert m.tool_call_id == "call_abc"
    assert m.tool_name == "list_tables"
