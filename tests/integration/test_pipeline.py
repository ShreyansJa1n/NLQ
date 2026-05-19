from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nl_db.llm.provider import ChatResult, Message
from nl_db.pipeline import Pipeline
from nl_db.validator import SQLValidationError


class CannedProvider:
    """Returns pre-programmed responses in order. Used to fake LLM calls."""

    name = "canned"
    model = "canned-1"

    def __init__(self, *responses: str) -> None:
        self._queue = list(responses)
        self.calls: list[list[Message]] = []
        self.call_kwargs: list[dict[str, object]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> ChatResult:
        self.calls.append(messages)
        self.call_kwargs.append(
            {"temperature": temperature, "max_output_tokens": max_output_tokens}
        )
        text = self._queue.pop(0) if self._queue else ""
        return ChatResult(text=text)


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    p = tmp_path / "sample.db"
    conn = sqlite3.connect(str(p))
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            amount_cents INTEGER NOT NULL,
            occurred_on TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        INSERT INTO users VALUES (1, 'alice'), (2, 'bob');
        INSERT INTO transactions VALUES
            (1, 1, 500, '2026-05-01'),
            (2, 1, 1200, '2026-05-15'),
            (3, 2, 300, '2026-05-10');
        """
    )
    conn.commit()
    conn.close()
    return p


def test_pipeline_end_to_end_simple_select(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nSELECT name FROM users ORDER BY id\n```",
        "Returns the name of every user.",
    )
    pipe = Pipeline(provider=provider, db_path=sample_db, max_rows=100)
    out = pipe.run("list every user's name")

    assert out.confirmed is True
    assert out.is_destructive is False
    assert out.auto_limit_applied is True
    assert "LIMIT" in out.sql_final.upper()
    assert out.paraphrase == "Returns the name of every user."
    assert out.result is not None
    assert out.result.columns == ("name",)
    assert out.result.rows == [("alice",), ("bob",)]


def test_pipeline_rejects_delete_without_allow_writes(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nDELETE FROM transactions\n```",
        "Deletes every transaction.",  # paraphrase shouldn't be reached
    )
    pipe = Pipeline(provider=provider, db_path=sample_db)
    with pytest.raises(SQLValidationError, match="destructive"):
        pipe.run("delete everything")


def test_pipeline_allows_delete_with_allow_writes(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nDELETE FROM transactions WHERE id = 99\n```",
        "Deletes the transaction with id 99 (none match).",
    )
    pipe = Pipeline(provider=provider, db_path=sample_db, paraphrase=True)
    out = pipe.run("delete transaction 99", allow_writes=True)
    assert out.is_destructive is True
    assert out.auto_limit_applied is False
    assert out.confirmed is True


def test_pipeline_skips_execution_when_user_declines(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nSELECT 1\n```",
        "Returns the number 1.",
    )
    pipe = Pipeline(provider=provider, db_path=sample_db)
    out = pipe.run("just one", confirm=lambda _sql, _para: False)
    assert out.confirmed is False
    assert out.result is None
    assert out.skipped_reason == "user declined to run the SQL"


def test_pipeline_paraphrase_disabled(sample_db: Path) -> None:
    provider = CannedProvider("```sql\nSELECT 1\n```")
    pipe = Pipeline(provider=provider, db_path=sample_db, paraphrase=False)
    out = pipe.run("just one")
    assert out.paraphrase is None
    # only one LLM call (no paraphrase pass)
    assert len(provider.calls) == 1


def test_pipeline_passes_temperature_and_max_tokens_through(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nSELECT 1\n```",
        "Returns the number 1.",
    )
    pipe = Pipeline(
        provider=provider,
        db_path=sample_db,
        temperature=0.7,
        max_output_tokens=256,
        paraphrase_temperature=0.3,
        paraphrase_max_output_tokens=64,
    )
    pipe.run("just one")
    assert provider.call_kwargs[0] == {"temperature": 0.7, "max_output_tokens": 256}
    assert provider.call_kwargs[1] == {"temperature": 0.3, "max_output_tokens": 64}


def test_pipeline_auto_limit_can_be_disabled(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nSELECT id FROM users\n```",
        "Returns user ids.",
    )
    pipe = Pipeline(provider=provider, db_path=sample_db, auto_limit=False)
    out = pipe.run("ids")
    assert out.auto_limit_applied is False
    assert "LIMIT" not in out.sql_final.upper()


def test_pipeline_num_few_shot_zero_omits_examples(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nSELECT 1\n```",
        "One.",
    )
    pipe = Pipeline(provider=provider, db_path=sample_db, num_few_shot=0)
    pipe.run("nothing useful")
    user_prompt = provider.calls[0][1].content
    assert "Example 1" not in user_prompt
    assert "Schema:" in user_prompt


def test_pipeline_num_few_shot_one_truncates_examples(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nSELECT 1\n```",
        "One.",
    )
    pipe = Pipeline(provider=provider, db_path=sample_db, num_few_shot=1)
    pipe.run("one example only")
    user_prompt = provider.calls[0][1].content
    assert "Example 1 question:" in user_prompt
    assert "Example 2 question:" not in user_prompt


# Three-state outcome tests -------------------------------------------------

def test_pipeline_cannot_answer_short_circuits(sample_db: Path) -> None:
    from nl_db.generator import CannotAnswer

    provider = CannedProvider(
        "CANNOT_ANSWER: This database has no information about employees."
    )
    pipe = Pipeline(provider=provider, db_path=sample_db)
    out = pipe.run("how many employees do we have?")

    assert out.state == "CANNOT_ANSWER"
    assert isinstance(out.outcome, CannotAnswer)
    assert "employees" in out.outcome.reason.lower()
    # Pipeline injects available_tables from the live schema
    assert out.outcome.available_tables == ("transactions", "users")
    # No SQL path was taken
    assert out.sql_final is None
    assert out.paraphrase is None
    assert out.result is None
    # Only one LLM call (no paraphrase, no SQL retry)
    assert len(provider.calls) == 1


def test_pipeline_clarify_short_circuits(sample_db: Path) -> None:
    from nl_db.generator import Clarify

    provider = CannedProvider(
        "CLARIFY: Do you mean spending in the current calendar month or the last 30 days?"
    )
    pipe = Pipeline(provider=provider, db_path=sample_db)
    out = pipe.run("how much did Alice spend recently?")

    assert out.state == "CLARIFY"
    assert isinstance(out.outcome, Clarify)
    assert "calendar month" in out.outcome.question
    assert out.sql_final is None
    assert out.result is None
    assert len(provider.calls) == 1


def test_pipeline_answer_state_property(sample_db: Path) -> None:
    from nl_db.generator import Answer

    provider = CannedProvider(
        "```sql\nSELECT id FROM users\n```",
        "Lists user ids.",
    )
    out = Pipeline(provider=provider, db_path=sample_db).run("list ids")
    assert out.state == "ANSWER"
    assert isinstance(out.outcome, Answer)
    assert out.outcome.sql.upper().startswith("SELECT")


def test_pipeline_history_appears_in_prompt(sample_db: Path) -> None:
    from nl_db.conversation import Conversation, Turn
    from nl_db.generator import Answer

    provider = CannedProvider(
        "```sql\nSELECT id FROM users\n```",
        "Lists user ids.",
    )
    history = Conversation()
    history.append(
        Turn(
            question="list users",
            outcome=Answer(sql="SELECT name FROM users;"),
            row_summary="name — 2 rows; first: Alice",
        )
    )
    pipe = Pipeline(provider=provider, db_path=sample_db)
    pipe.run("now their ids", history=history)
    user_prompt = provider.calls[0][1].content
    assert "Conversation so far:" in user_prompt
    assert "Q: list users" in user_prompt
    assert "name — 2 rows" in user_prompt


def test_pipeline_no_history_no_conversation_block(sample_db: Path) -> None:
    provider = CannedProvider(
        "```sql\nSELECT id FROM users\n```",
        "Lists user ids.",
    )
    Pipeline(provider=provider, db_path=sample_db).run("list ids")
    user_prompt = provider.calls[0][1].content
    assert "Conversation so far:" not in user_prompt
