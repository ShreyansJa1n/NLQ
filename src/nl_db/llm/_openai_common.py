"""Shared OpenAI chat-completions wire-format helpers.

Both OpenAIProvider (against api.openai.com) and OpenAICompatibleProvider
(against any /v1/chat/completions endpoint) speak the same wire format.
This module centralises:
  - building the messages array (handling role='tool' loop-back)
  - building the tools array from our ToolDef tuple
  - parsing tool_calls back into our ToolCall shape
  - mapping shim errors to ToolsNotSupportedError when tools were requested
"""
from __future__ import annotations

import json
from typing import Any

from .provider import (
    ChatResult,
    Message,
    ToolCall,
    ToolDef,
    ToolsNotSupportedError,
)


def build_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate our Message list into OpenAI chat-completions format.

    - role='tool' Messages become {"role": "tool", "tool_call_id": "...", "content": "..."}
    - other roles map 1:1 with string content
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": m.content,
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def build_openai_tools(tools: tuple[ToolDef, ...]) -> list[dict[str, Any]]:
    """Translate our ToolDef tuple into OpenAI's `tools=[{type: function, ...}]` shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def parse_openai_tool_calls(message: Any) -> tuple[ToolCall, ...]:
    """Extract tool_calls from an openai SDK message object.

    openai returns each tool call as an object with `.id`, `.type='function'`,
    and `.function.name` / `.function.arguments` (the arguments are a
    JSON-encoded string, not a parsed dict — unlike Anthropic).
    """
    raw = getattr(message, "tool_calls", None) or []
    out: list[ToolCall] = []
    for tc in raw:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        args_raw = getattr(fn, "arguments", "") or ""
        try:
            args = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            args = {"_raw": args_raw}
        out.append(
            ToolCall(
                id=str(getattr(tc, "id", "")),
                name=str(getattr(fn, "name", "")),
                arguments=args,
            )
        )
    return tuple(out)


def call_openai_chat(
    client: Any,
    *,
    model: str,
    messages: list[Message],
    temperature: float,
    max_output_tokens: int,
    tools: tuple[ToolDef, ...] | None,
) -> ChatResult:
    """One round-trip against an openai-compatible /v1/chat/completions endpoint.

    Raises ToolsNotSupportedError if the endpoint rejects the tools= argument
    (e.g. a shim that doesn't implement tool-calling returns 400).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": build_openai_messages(messages),
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if tools:
        kwargs["tools"] = build_openai_tools(tools)

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        # Only translate to ToolsNotSupportedError when the caller actually
        # asked for tools — if tools=None and we still got an error, it's
        # a real failure (auth, network, etc.) the caller needs to see.
        if tools and _looks_like_tools_unsupported(e):
            raise ToolsNotSupportedError(
                f"Endpoint rejected tools= argument: {e}"
            ) from e
        raise

    choice = response.choices[0]
    text = choice.message.content or ""
    tool_calls = parse_openai_tool_calls(choice.message)
    usage = getattr(response, "usage", None)
    return ChatResult(
        text=text,
        input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        tool_calls=tool_calls,
        stop_reason=getattr(choice, "finish_reason", None),
        provider_meta={
            "model": model,
            "finish_reason": getattr(choice, "finish_reason", None),
        },
    )


def _looks_like_tools_unsupported(exc: Exception) -> bool:
    """Heuristic: does this exception look like the endpoint rejected the
    tools= argument as opposed to some other 4xx/5xx?

    We pattern-match on the message because the openai SDK's exception
    hierarchy varies across versions and a strict isinstance check would
    miss subclasses we don't know about. Conservative — when in doubt we
    let the original exception propagate.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    # The most common cases: BadRequestError with a mention of 'tool'/'function',
    # or APIStatusError variants saying the same.
    return name in (
        "BadRequestError",
        "UnprocessableEntityError",
        "APIStatusError",
    ) and any(
        kw in msg for kw in ("tool", "function", "unsupported", "unknown parameter")
    )
