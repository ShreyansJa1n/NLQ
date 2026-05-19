from __future__ import annotations

from dataclasses import dataclass

from ..llm.provider import Message
from ..schema.base import Schema, render_for_prompt
from .examples import FewShotExample, few_shot_for
from .system import system_prompt


@dataclass
class BuiltPrompt:
    messages: list[Message]
    approx_tokens: int


def _approx_tokens(text: str) -> int:
    """Rough char/4 heuristic — good enough for budget warnings."""
    return max(1, len(text) // 4)


def build_sql_prompt(
    schema: Schema,
    question: str,
    *,
    examples: tuple[FewShotExample, ...] | None = None,
    max_tokens_hint: int | None = None,
) -> BuiltPrompt:
    """Assemble system + schema + few-shot + question into a chat message list.

    If `max_tokens_hint` is set and the assembled prompt exceeds it, the
    return still includes everything — the caller decides whether to warn,
    truncate, or proceed.
    """
    examples = examples if examples is not None else few_shot_for(schema.dialect)

    schema_block = render_for_prompt(schema)

    user_parts = [f"Schema:\n{schema_block}"]
    if examples:
        ex_lines = []
        for i, ex in enumerate(examples, start=1):
            ex_lines.append(f"Example {i} question: {ex.question}")
            ex_lines.append(f"Example {i} SQL:\n```sql\n{ex.sql}\n```")
        user_parts.append("\n\n".join(ex_lines))
    user_parts.append(f"Question: {question}\n\nSQL:")

    user_content = "\n\n".join(user_parts)

    messages = [
        Message(role="system", content=system_prompt(schema.dialect)),
        Message(role="user", content=user_content),
    ]
    total = sum(_approx_tokens(m.content) for m in messages)
    return BuiltPrompt(messages=messages, approx_tokens=total)


def exceeds_budget(prompt: BuiltPrompt, max_tokens: int) -> bool:
    return prompt.approx_tokens > max_tokens
