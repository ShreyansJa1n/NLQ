from __future__ import annotations

import re

from .llm.provider import LLMProvider
from .prompts.builder import BuiltPrompt


class SQLExtractionError(ValueError):
    """Raised when no SQL statement can be extracted from an LLM response."""


_FENCE_PATTERN = re.compile(
    r"```(?:sql|sqlite|postgres|mysql)?\s*(.+?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_sql(text: str) -> str:
    """Pull the first SQL statement out of an LLM response.

    Handles:
    - markdown fences with or without a language tag
    - leading/trailing prose (warns by ignoring, not failing)
    - multiple statements (returns the first)

    Raises SQLExtractionError if nothing usable is found.
    """
    candidate: str | None = None

    fence_match = _FENCE_PATTERN.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        stripped = text.strip()
        if stripped:
            candidate = stripped

    if not candidate:
        raise SQLExtractionError("LLM response contained no SQL")

    candidate = candidate.split(";")[0].strip()
    if not candidate:
        raise SQLExtractionError("LLM response contained only an empty statement")

    return candidate + ";"


def generate_sql(
    provider: LLMProvider,
    prompt: BuiltPrompt,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
) -> str:
    """Run the prompt through the provider and extract a single SQL statement."""
    result = provider.chat(
        prompt.messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return extract_sql(result.text)
