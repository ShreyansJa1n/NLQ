from __future__ import annotations

import pytest

from nl_db.generator import SQLExtractionError, extract_sql
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
