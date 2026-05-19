from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass
class ChatResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_meta: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Provider-agnostic chat interface.

    Pipeline code MUST go through this Protocol — never import vendor SDKs
    directly. Add new backends via `nl_db.llm.registry.build_provider`.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> ChatResult: ...
