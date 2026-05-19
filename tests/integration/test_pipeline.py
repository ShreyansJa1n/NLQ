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

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> ChatResult:
        self.calls.append(messages)
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
