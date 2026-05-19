from __future__ import annotations

from nl_db.conversation import Conversation, Turn, summarize_rows
from nl_db.generator import Answer, CannotAnswer, Clarify


def test_empty_conversation_renders_empty_context() -> None:
    assert Conversation().to_prompt_context() == ""
    assert Conversation().is_empty() is True


def test_append_grows_turns() -> None:
    c = Conversation()
    c.append(Turn(question="q1", outcome=Answer(sql="SELECT 1;")))
    c.append(Turn(question="q2", outcome=Answer(sql="SELECT 2;")))
    assert len(c.turns) == 2
    assert c.is_empty() is False


def test_to_prompt_context_renders_answer_with_row_summary() -> None:
    c = Conversation()
    c.append(
        Turn(
            question="list users",
            outcome=Answer(sql="SELECT name FROM users;"),
            row_summary="name — 2 rows; first: Alice",
        )
    )
    text = c.to_prompt_context()
    assert "Q: list users" in text
    assert "SQL: SELECT name FROM users;" in text
    assert "Result: name — 2 rows; first: Alice" in text


def test_to_prompt_context_renders_cannot_answer() -> None:
    c = Conversation()
    c.append(
        Turn(
            question="how many employees?",
            outcome=CannotAnswer(reason="no employees table", available_tables=("users",)),
        )
    )
    text = c.to_prompt_context()
    assert "[Could not answer: no employees table]" in text


def test_to_prompt_context_renders_clarify() -> None:
    c = Conversation()
    c.append(
        Turn(
            question="show me sales",
            outcome=Clarify(question="Last month or last 30 days?"),
        )
    )
    text = c.to_prompt_context()
    assert "[Asked for clarification: Last month or last 30 days?]" in text


def test_to_prompt_context_bounds_by_max_turns() -> None:
    c = Conversation()
    for i in range(10):
        c.append(Turn(question=f"q{i}", outcome=Answer(sql=f"SELECT {i};")))
    text = c.to_prompt_context(max_turns=3)
    # only the last three turns should appear
    assert "Q: q7" in text
    assert "Q: q9" in text
    assert "Q: q0" not in text
    assert "Q: q6" not in text


def test_summarize_rows_empty() -> None:
    assert summarize_rows(("name",), []) == "name — 0 rows"


def test_summarize_rows_one_row() -> None:
    summary = summarize_rows(("name", "age"), [("Alice", 30)])
    assert "1 row" in summary
    assert "Alice" in summary


def test_summarize_rows_many_rows() -> None:
    rows = [(f"name{i}", i) for i in range(50)]
    summary = summarize_rows(("name", "age"), rows)
    assert "50 rows" in summary
    assert "name0" in summary  # first row included


def test_summarize_rows_truncates_long_first_row() -> None:
    long_string = "x" * 500
    summary = summarize_rows(("data",), [(long_string,)], max_chars=50)
    assert "…" in summary
    assert len(summary) < len(long_string)
