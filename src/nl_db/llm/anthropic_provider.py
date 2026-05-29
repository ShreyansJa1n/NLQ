from __future__ import annotations

from typing import Any, cast

import anthropic

from .provider import ChatResult, Message


class AnthropicProvider:
    name = "anthropic"
    # All current Claude 3.x and 4.x models support tool-calling.
    supports_tools: bool | None = True

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
        tools: tuple[Any, ...] | None = None,
    ) -> ChatResult:
        # Tools are wired through in the next commit (anthropic tool-calling).
        # Until then, accepting the kwarg keeps the Protocol consistent.
        if tools:
            from .provider import ToolsNotSupportedError

            raise ToolsNotSupportedError(
                "Anthropic tool-calling not yet wired in this provider."
            )
        system_parts = [m.content for m in messages if m.role == "system"]
        convo = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text_parts.append(cast(str, getattr(block, "text", "")))
        text = "".join(text_parts)

        usage = getattr(response, "usage", None)
        return ChatResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            provider_meta={
                "model": self._model,
                "stop_reason": getattr(response, "stop_reason", None),
            },
        )
