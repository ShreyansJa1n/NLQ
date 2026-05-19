from __future__ import annotations

from typing import Any

import anthropic

from .provider import ChatResult, Message


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str, client: Any | None = None) -> None:
        self._model = model
        self._client = client or anthropic.Anthropic(api_key=api_key)

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
        system_parts = [m.content for m in messages if m.role == "system"]
        convo = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            system="\n\n".join(system_parts) if system_parts else anthropic.NOT_GIVEN,
            messages=convo,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return ChatResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            provider_meta={"model": self._model, "stop_reason": getattr(response, "stop_reason", None)},
        )
