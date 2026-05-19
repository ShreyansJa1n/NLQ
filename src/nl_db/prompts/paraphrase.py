from __future__ import annotations

from ..llm.provider import LLMProvider, Message
from .system import PARAPHRASE_SYSTEM


def paraphrase_sql(provider: LLMProvider, sql: str) -> str:
    """Ask the LLM to NL-explain a SQL statement in one short sentence.

    Schema is intentionally NOT re-sent — paraphrase reads the SQL itself.
    """
    result = provider.chat(
        [
            Message(role="system", content=PARAPHRASE_SYSTEM),
            Message(role="user", content=f"SQL:\n```sql\n{sql}\n```"),
        ],
        temperature=0.0,
        max_output_tokens=128,
    )
    return result.text.strip()
