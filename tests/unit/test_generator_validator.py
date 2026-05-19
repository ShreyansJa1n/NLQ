from __future__ import annotations

import pytest

from nl_db.generator import (
    Answer,
    CannotAnswer,
    Clarify,
    SQLExtractionError,
    extract_sql,
    parse_outcome,
)
from nl_db.validator import SQLValidationError, validate_sql

# extract_sql ---------------------------------------------------------------

def test_extract_from_fenced_sql_block() -> None:
    text = "Here you go:\n```sql\nSELECT * FROM users\n```\n"
    assert extract_sql(text) == "SELECT * FROM users;"


def test_extract_from_unfenced_response() -> None:
    text = "SELECT id FROM users WHERE id = 1;\n"
    assert extract_sql(text) == "SELECT id FROM users WHERE id = 1;"


def test_extract_handles_no_language_tag_fence() -> None:
    text = "```\nSELECT 1\n```"
    assert extract_sql(text) == "SELECT 1;"


def test_extract_strips_trailing_statements() -> None:
    text = "```sql\nSELECT 1; SELECT 2;\n```"
    assert extract_sql(text) == "SELECT 1;"


def test_extract_empty_response_raises() -> None:
    with pytest.raises(SQLExtractionError):
        extract_sql("")


def test_extract_only_fence_raises() -> None:
    with pytest.raises(SQLExtractionError):
        extract_sql("```sql\n   \n```")


# parse_outcome ------------------------------------------------------------

def test_parse_outcome_fenced_sql_is_answer() -> None:
    outcome = parse_outcome("```sql\nSELECT 1\n```")
    assert isinstance(outcome, Answer)
    assert outcome.sql == "SELECT 1;"


def test_parse_outcome_unfenced_sql_is_answer() -> None:
    outcome = parse_outcome("SELECT 1;")
    assert isinstance(outcome, Answer)


def test_parse_outcome_cannot_answer_sentinel() -> None:
    outcome = parse_outcome(
        "CANNOT_ANSWER: This database has no information about employees."
    )
    assert isinstance(outcome, CannotAnswer)
    assert "employees" in outcome.reason
    # available_tables is filled in by the Pipeline, not the parser
    assert outcome.available_tables == ()


def test_parse_outcome_cannot_answer_case_insensitive() -> None:
    outcome = parse_outcome("cannot_answer: no products table here.")
    assert isinstance(outcome, CannotAnswer)


def test_parse_outcome_cannot_answer_in_fenced_block() -> None:
    # Some models wrap sentinel responses in a code fence — strip and detect.
    outcome = parse_outcome("```\nCANNOT_ANSWER: nothing relevant.\n```")
    assert isinstance(outcome, CannotAnswer)
    assert "nothing relevant" in outcome.reason


def test_parse_outcome_clarify_sentinel() -> None:
    outcome = parse_outcome(
        "CLARIFY: Do you mean last calendar month or the last 30 days?"
    )
    assert isinstance(outcome, Clarify)
    assert "calendar month" in outcome.question


def test_parse_outcome_clarify_with_leading_whitespace() -> None:
    outcome = parse_outcome("   CLARIFY: Which user?")
    assert isinstance(outcome, Clarify)
    assert outcome.question == "Which user?"


def test_parse_outcome_garbage_raises_via_sql_path() -> None:
    # If it's not a sentinel and not extractable SQL, raise from the SQL path.
    # Empty string definitely fails.
    with pytest.raises(SQLExtractionError):
        parse_outcome("")


# validate_sql --------------------------------------------------------------

def test_validate_simple_select_injects_limit() -> None:
    result = validate_sql("SELECT * FROM users", max_rows=100)
    assert result.is_destructive is False
    assert result.auto_limit_applied is True
    assert "LIMIT 100" in result.sql.upper()


def test_validate_select_with_existing_limit_left_alone() -> None:
    result = validate_sql("SELECT * FROM users LIMIT 5", max_rows=100)
    assert result.auto_limit_applied is False
    assert "LIMIT 5" in result.sql.upper()


def test_validate_no_limit_when_max_rows_none() -> None:
    result = validate_sql("SELECT * FROM users", max_rows=None)
    assert result.auto_limit_applied is False
    assert "LIMIT" not in result.sql.upper()


def test_validate_rejects_insert_without_allow_writes() -> None:
    with pytest.raises(SQLValidationError, match="destructive"):
        validate_sql("INSERT INTO users (id) VALUES (1)")


def test_validate_rejects_update_without_allow_writes() -> None:
    with pytest.raises(SQLValidationError, match="destructive"):
        validate_sql("UPDATE users SET name = 'x'")


def test_validate_rejects_delete_without_allow_writes() -> None:
    with pytest.raises(SQLValidationError, match="destructive"):
        validate_sql("DELETE FROM users")


def test_validate_allows_destructive_with_flag() -> None:
    result = validate_sql("DELETE FROM users", allow_writes=True, max_rows=100)
    assert result.is_destructive is True
    # auto-limit not injected for destructive
    assert "LIMIT" not in result.sql.upper()


def test_validate_rejects_multiple_statements() -> None:
    with pytest.raises(SQLValidationError, match="one statement"):
        validate_sql("SELECT 1; SELECT 2;")


def test_validate_rejects_garbage() -> None:
    with pytest.raises(SQLValidationError):
        validate_sql("this is not sql at all !! @#$")


def test_validate_select_with_cte_and_aggregate_no_limit_still_capped() -> None:
    sql = "WITH t AS (SELECT 1 AS x) SELECT SUM(x) FROM t"
    result = validate_sql(sql, max_rows=10)
    # SUM-only aggregate returns 1 row anyway; we still inject LIMIT for safety
    assert "LIMIT 10" in result.sql.upper()
