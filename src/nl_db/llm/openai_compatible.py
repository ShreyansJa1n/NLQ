from __future__ import annotations

from typing import Any

import openai

from ._openai_common import call_openai_chat
from .provider import ChatResult, Message, ToolDef


class OpenAICompatibleProvider:
    """For any endpoint that speaks the OpenAI chat-completions wire format.

    Examples: Ollama (`http://localhost:11434/v1`), vLLM, LM Studio,
    or the Apple Intelligence Swift HTTP shim used by this project.

    Many local servers don't require a real API key — pass a placeholder
    if the server ignores it.
    """

    name = "openai_compatible"
    # Tool-calling support depends entirely on the shim + model. Ollama
    # implements it for some models; Apple Intelligence shims typically
    # don't. None means: the lazy-schema orchestrator should attempt tools
    # and fall back to schema injection if the shim rejects them or the
    # model ignores them.
    supports_tools: bool | None = None

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._client = client or openai.OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
        )

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        tools: tuple[ToolDef, ...] | None = None,
    ) -> ChatResult:
        # call_openai_chat translates shim rejections of tools= into
        # ToolsNotSupportedError for the orchestrator to catch.
        return call_openai_chat(
            self._client,
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
        )
