from __future__ import annotations

from typing import Any, cast

import openai

from .provider import ChatResult, Message


class OpenAIProvider:
    name = "openai"
    # All current GPT-4* / GPT-4o* / GPT-3.5-turbo support tool-calling.
    supports_tools: bool | None = True

    def __init__(self, model: str, api_key: str, client: Any | None = None) -> None:
        self._model = model
        self._client = client or openai.OpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        tools: tuple[Any, ...] | None = None,
    ) -> ChatResult:
        # Tools are wired through in the next commit (OpenAI tool-calling).
        if tools:
            from .provider import ToolsNotSupportedError

            raise ToolsNotSupportedError(
                "OpenAI tool-calling not yet wired in this provider."
            )
        wire = cast(
            Any,
            [{"role": m.role, "content": m.content} for m in messages],
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=wire,
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
            provider_meta={"model": self._model, "finish_reason": choice.finish_reason},
        )
