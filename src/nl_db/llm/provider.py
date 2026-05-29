from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    """A chat message.

    For role="tool" messages: `content` is the JSON-serialized tool result,
    `tool_call_id` identifies the call we're responding to, `tool_name`
    is the name of the tool that produced this result. (Providers use one
    or the other; we carry both so each backend can pick.)
    """

    role: Role
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class ToolDef:
    """Provider-agnostic tool definition.

    Each provider translates this into its native wire format:
      - Anthropic: `tools=[{"name", "description", "input_schema"}]`
      - OpenAI:    `tools=[{"type": "function", "function": {...}}]`
    """

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema for the tool's input


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to invoke a tool.

    `id` is the provider's identifier for this call (used to match up the
    tool_result when we reply). `name` and `arguments` describe what to run.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_meta: dict[str, Any] = field(default_factory=dict)
    # When non-empty, the model is asking us to run tools. The caller should
    # execute each ToolCall, append a corresponding role="tool" Message to
    # the conversation, and call chat() again. The loop ends when the model
    # returns a ChatResult with no tool_calls.
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None


class ToolsNotSupportedError(RuntimeError):
    """Raised when the caller asked for tools and we know — or just learned —
    that the configured provider/model doesn't actually support them.

    The lazy-schema orchestrator catches this and falls back to schema
    injection. Surfaces clearly in the Debug expander.
    """


class LLMProvider(Protocol):
    """Provider-agnostic chat interface.

    Pipeline code MUST go through this Protocol — never import vendor SDKs
    directly. Add new backends via `nl_db.llm.registry.build_provider`.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def supports_tools(self) -> bool | None:
        """Tool-calling support for this provider/model.

        Three-state:
          - True  → known to support tools (Anthropic, OpenAI)
          - False → known NOT to support tools (rare for current providers,
                    but reserved for known-incompatible shim configs)
          - None  → unknown; the lazy-schema orchestrator will attempt
                    tools and fall back to schema injection on failure
                    (typical for openai_compatible, where capability
                    depends on the shim + model in use)

        Static — no probe calls at construction time.
        """
        ...

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        tools: tuple[ToolDef, ...] | None = None,
    ) -> ChatResult:
        """Run one round-trip against the LLM.

        When `tools` is provided and the model wants to use one, the result
        carries `tool_calls` and `text` is typically empty (or contains a
        pre-tool-use commentary block, depending on the provider). The
        caller is expected to run a tool-use loop.

        Should raise `ToolsNotSupportedError` if the provider/model rejects
        the tools argument outright (e.g. an openai-compatible shim returns
        a 400 because it doesn't implement the spec). Subtler "model accepts
        the tools= argument but ignores them" cases surface as a result
        with empty tool_calls — the orchestrator detects this separately.
        """
        ...
