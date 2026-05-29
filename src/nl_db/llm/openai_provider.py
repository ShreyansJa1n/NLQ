from __future__ import annotations

from typing import Any

import openai

from ._openai_common import call_openai_chat
from .provider import ChatResult, Message, ToolDef


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
        tools: tuple[ToolDef, ...] | None = None,
    ) -> ChatResult:
        return call_openai_chat(
            self._client,
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=tools,
        )
