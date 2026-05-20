from __future__ import annotations

from ..llm.provider import LLMProvider, Message
from .system import PARAPHRASE_SYSTEM


def paraphrase_sql(
    provider: LLMProvider,
    sql: str,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 512,
) -> str:
    """Ask the LLM to NL-explain a SQL statement in one short sentence.

    Schema is intentionally NOT re-sent — paraphrase reads the SQL itself.
    """
    result = provider.chat(
        [
            Message(role="system", content=PARAPHRASE_SYSTEM),
            Message(role="user", content=f"SQL:\n```sql\n{sql}\n```"),
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return result.text.strip()
