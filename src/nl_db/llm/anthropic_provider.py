from __future__ import annotations

from typing import Any, cast

import anthropic

from .provider import ChatResult, Message, ToolCall, ToolDef


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
        tools: tuple[ToolDef, ...] | None = None,
    ) -> ChatResult:
        system_parts = [m.content for m in messages if m.role == "system"]
        convo = _build_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(cast(str, getattr(block, "text", "")))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=cast(str, getattr(block, "id", "")),
                        name=cast(str, getattr(block, "name", "")),
                        arguments=cast(dict[str, Any], getattr(block, "input", {}) or {}),
                    )
                )
        text = "".join(text_parts)

        usage = getattr(response, "usage", None)
        stop_reason = getattr(response, "stop_reason", None)
        return ChatResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            tool_calls=tuple(tool_calls),
            stop_reason=stop_reason,
            provider_meta={
                "model": self._model,
                "stop_reason": stop_reason,
            },
        )


def _build_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate our Message list into Anthropic's content-block format.

    Anthropic doesn't have a 'tool' role — tool results come back as a
    `user` message whose content includes one or more `tool_result` blocks
    referencing the original `tool_use_id`. Group consecutive role="tool"
    messages into a single user message so the wire format stays compact.

    Plain user/assistant text messages translate 1:1 with string content.
    """
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }
            )
            continue
        flush_tool_results()
        out.append({"role": m.role, "content": m.content})
    flush_tool_results()
    return out
