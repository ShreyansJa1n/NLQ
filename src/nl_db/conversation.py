"""Multi-turn conversation state.

A `Conversation` is an ordered list of past turns. Each turn captures the
user's question, the generation outcome (Answer / CannotAnswer / Clarify),
and — when the outcome was Answer — a tiny summary of the result rows.

When we build the prompt for the next question, we include a compact textual
rendering of recent turns so the LLM can resolve follow-ups like "now group
by region" or "and for last month".

Design notes:
- The schema is always the primary part of the prompt. History is bounded
  by `max_turns` (default 5) so it can't crowd out the schema.
- Result rows are not included verbatim — that risks PII and token bloat.
  Instead we include a one-line "row_summary" (column names + row count +
  optionally the first row).
- This is in-memory only. The MCP server keeps a `dict[conversation_id, Conversation]`
  for the lifetime of the process; the Streamlit playground keeps one in
  `st.session_state`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .generator import Answer, CannotAnswer, Clarify, GenerationOutcome


@dataclass(frozen=True)
class Turn:
    question: str
    outcome: GenerationOutcome
    row_summary: str | None = None  # one-line summary of the Answer's result rows


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)

    def append(self, turn: Turn) -> None:
        self.turns.append(turn)

    def is_empty(self) -> bool:
        return not self.turns

    def to_prompt_context(self, *, max_turns: int = 5) -> str:
        """Render the most recent `max_turns` turns as a compact string for prompt injection.

        Format (one turn per block, oldest first):
            Q: <question>
            <one of>
              SQL: <sql>            (Answer)
              Result: <row_summary>  (Answer with summary)
            or:
              [Could not answer: <reason>]   (CannotAnswer)
            or:
              [Asked for clarification: <q>] (Clarify)
        """
        if not self.turns:
            return ""
        recent = self.turns[-max_turns:]
        blocks: list[str] = []
        for t in recent:
            lines = [f"Q: {t.question}"]
            if isinstance(t.outcome, Answer):
                lines.append(f"SQL: {t.outcome.sql.strip()}")
                if t.row_summary:
                    lines.append(f"Result: {t.row_summary}")
            elif isinstance(t.outcome, CannotAnswer):
                lines.append(f"[Could not answer: {t.outcome.reason}]")
            elif isinstance(t.outcome, Clarify):
                lines.append(f"[Asked for clarification: {t.outcome.question}]")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def summarize_rows(
    columns: tuple[str, ...] | list[str],
    rows: list[tuple[object, ...]] | list[list[object]],
    *,
    max_chars: int = 200,
) -> str:
    """One-line summary of a result set, safe to include in a prompt.

    Includes column names, row count, and the first row if available.
    """
    cols = ", ".join(columns) if columns else "(no columns)"
    n = len(rows)
    if n == 0:
        return f"{cols} — 0 rows"
    first = ", ".join(str(v) for v in rows[0])
    if len(first) > max_chars:
        first = first[:max_chars] + "…"
    if n == 1:
        return f"{cols} — 1 row: {first}"
    return f"{cols} — {n} rows; first: {first}"
