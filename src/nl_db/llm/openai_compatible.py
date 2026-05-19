from __future__ import annotations

from typing import Any

import openai

from .provider import ChatResult, Message


class OpenAICompatibleProvider:
    """For any endpoint that speaks the OpenAI chat-completions wire format.

    Examples: Ollama (`http://localhost:11434/v1`), vLLM, LM Studio,
    or the Apple Intelligence Swift HTTP shim used by this project.

    Many local servers don't require a real API key — pass a placeholder
    if the server ignores it.
    """

    name = "openai_compatible"

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
    ) -> ChatResult:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        return ChatResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            provider_meta={
                "model": self._model,
                "finish_reason": getattr(choice, "finish_reason", None),
            },
        )
