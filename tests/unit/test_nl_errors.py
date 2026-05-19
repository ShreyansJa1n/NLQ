from __future__ import annotations

import sqlite3

from nl_db.generator import SQLExtractionError
from nl_db.nl_errors import humanize
from nl_db.validator import SQLValidationError


def test_humanize_destructive_sql_validation_error() -> None:
    msg = humanize(SQLValidationError("Refusing to run destructive SQL"))
    assert "change the database" in msg.lower()
    assert "--allow-writes" in msg


def test_humanize_multi_statement_validation_error() -> None:
    msg = humanize(SQLValidationError("Expected exactly one statement, got 2"))
    assert "more than one statement" in msg.lower()


def test_humanize_parse_error_validation_error() -> None:
    msg = humanize(SQLValidationError("SQL parse error: bad syntax"))
    assert "valid SQL" in msg


def test_humanize_sql_extraction_error() -> None:
    msg = humanize(SQLExtractionError("nothing usable"))
    assert "didn't return a query" in msg.lower() or "couldn't read" in msg.lower()


def test_humanize_no_such_table() -> None:
    msg = humanize(sqlite3.OperationalError("no such table: employees"))
    assert "doesn't have a table named" in msg
    assert "`employees`" in msg


def test_humanize_no_such_column() -> None:
    msg = humanize(sqlite3.OperationalError("no such column: foo"))
    assert "doesn't have a column named" in msg
    assert "`foo`" in msg


def test_humanize_syntax_error() -> None:
    msg = humanize(sqlite3.OperationalError("near \"FROM\": syntax error"))
    assert "syntax error" in msg.lower()


def test_humanize_interrupted_query() -> None:
    msg = humanize(sqlite3.OperationalError("interrupted"))
    assert "too long" in msg.lower() or "cancelled" in msg.lower()


def test_humanize_not_found_error_class_name() -> None:
    # We pattern-match on class name to avoid importing vendor SDKs eagerly.
    class NotFoundError(Exception):
        pass

    msg = humanize(NotFoundError("Not Found"))
    assert "404" in msg
    assert "model name" in msg.lower()


def test_humanize_auth_error_class_name() -> None:
    class AuthenticationError(Exception):
        pass

    msg = humanize(AuthenticationError("invalid api key"))
    assert "API key" in msg or "api key" in msg.lower()


def test_humanize_connection_error_class_name() -> None:
    class APIConnectionError(Exception):
        pass

    msg = humanize(APIConnectionError("connection refused"))
    assert "couldn't reach" in msg.lower()


def test_humanize_generic_fallback() -> None:
    msg = humanize(RuntimeError("some weird thing"))
    assert "Something went wrong" in msg
    assert "RuntimeError" in msg
