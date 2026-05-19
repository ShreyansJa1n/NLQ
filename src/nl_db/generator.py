from __future__ import annotations

import re
from dataclasses import dataclass

from .llm.provider import LLMProvider
from .prompts.builder import BuiltPrompt


class SQLExtractionError(ValueError):
    """Raised when an LLM response looks like SQL but can't be extracted."""


@dataclass(frozen=True)
class Answer:
    """The LLM produced executable SQL."""

    sql: str


@dataclass(frozen=True)
class CannotAnswer:
    """The LLM determined the database can't answer this question.

    `reason` is a plain-English sentence from the LLM. `available_tables`
    is injected by the Pipeline from the live schema and lets the caller
    suggest alternatives without re-fetching the schema.
    """

    reason: str
    available_tables: tuple[str, ...] = ()


@dataclass(frozen=True)
class Clarify:
    """The LLM needs more information before it can generate SQL."""

    question: str


GenerationOutcome = Answer | CannotAnswer | Clarify


_FENCE_PATTERN = re.compile(
    r"```(?:sql|sqlite|postgres|mysql)?\s*(.+?)```",
    re.DOTALL | re.IGNORECASE,
)
_CANNOT_PATTERN = re.compile(
    r"^\s*CANNOT_ANSWER\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_CLARIFY_PATTERN = re.compile(
    r"^\s*CLARIFY\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _strip_fence(text: str) -> str:
    """If the text is wrapped in a single fenced block, return the inner content.

    Used so sentinel detection still works when the LLM wraps its sentinel
    response in a code fence (some models do this reflexively).
    """
    stripped = text.strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return text
    # remove leading fence (optionally with a language tag) and trailing fence
    inner = re.sub(r"^```(?:[a-zA-Z]+)?\s*", "", stripped)
    inner = re.sub(r"\s*```$", "", inner)
    return inner


def parse_outcome(text: str) -> GenerationOutcome:
    """Parse an LLM response into one of the three generation outcomes.

    Resolution order:
      1. If the (possibly unfenced) body starts with `CANNOT_ANSWER:` → CannotAnswer
      2. Same for `CLARIFY:` → Clarify
      3. Otherwise try to extract SQL → Answer

    `available_tables` is left empty on CannotAnswer here — the Pipeline fills
    it in from the live schema since the generator has no schema context.
    """
    body = _strip_fence(text)

    m = _CANNOT_PATTERN.match(body)
    if m:
        return CannotAnswer(reason=m.group(1).strip())

    m = _CLARIFY_PATTERN.match(body)
    if m:
        return Clarify(question=m.group(1).strip())

    return Answer(sql=extract_sql(text))


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


def generate_outcome(
    provider: LLMProvider,
    prompt: BuiltPrompt,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
) -> GenerationOutcome:
    """Call the provider, parse the response into a GenerationOutcome."""
    result = provider.chat(
        prompt.messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return parse_outcome(result.text)


def generate_sql(
    provider: LLMProvider,
    prompt: BuiltPrompt,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
) -> str:
    """Back-compat shim: call generate_outcome and return SQL if the outcome
    is Answer, otherwise raise SQLExtractionError. New code should call
    generate_outcome directly.
    """
    outcome = generate_outcome(
        provider, prompt, temperature=temperature, max_output_tokens=max_output_tokens
    )
    if isinstance(outcome, Answer):
        return outcome.sql
    raise SQLExtractionError(
        f"LLM did not produce SQL (outcome: {type(outcome).__name__})"
    )
