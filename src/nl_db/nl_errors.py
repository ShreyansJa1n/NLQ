"""Translate raw exceptions into plain-English sentences for end users.

The pipeline still raises raw exceptions — only the outer surfaces (CLI, UI,
MCP server wrappers) humanize them before showing the user. That keeps the
internals debuggable while making the surfaces friendly.

`humanize()` is dispatch-by-message-and-type. We don't import every possible
vendor SDK exception class at module import time (anthropic, openai, httpx,
sqlite3, sqlglot, etc. would balloon import cost); instead we pattern-match
on the class name and the string representation. That works because we're
producing a one-sentence summary, not a structured error.
"""
from __future__ import annotations

import sqlite3

from .generator import SQLExtractionError
from .validator import SQLValidationError


def humanize(exc: BaseException) -> str:
    """Return a one-sentence plain-English description of an exception.

    Designed for end users — no stack traces, no SQL jargon, no vendor SDK
    class names. If the exception is unrecognised, returns a generic
    "something went wrong" message that includes the type name so it can
    still be diagnosed from logs.
    """
    name = type(exc).__name__
    msg = str(exc).strip()

    # nl-db-internal exceptions ------------------------------------------------
    if isinstance(exc, SQLValidationError):
        # Validator messages are already user-facing-ish but include SQL jargon.
        if "destructive" in msg.lower():
            return (
                "That request would change the database, which isn't allowed by "
                "default. Re-run with --allow-writes if you really want to."
            )
        if "one statement" in msg.lower() or "Expected exactly one" in msg:
            return "The query I generated had more than one statement. Please rephrase."
        if "parse error" in msg.lower():
            return "The query I generated wasn't valid SQL. Please rephrase or try again."
        return f"I can't run that query: {msg}"

    if isinstance(exc, SQLExtractionError):
        return (
            "The model didn't return a query I could read. Try rephrasing the "
            "question — shorter is often better."
        )

    # SQLite execution errors --------------------------------------------------
    if isinstance(exc, sqlite3.OperationalError):
        low = msg.lower()
        if "no such table" in low:
            tbl = _extract_quoted_or_word(msg, "no such table:") or "that table"
            return f"The database doesn't have a table named {tbl}."
        if "no such column" in low:
            col = _extract_quoted_or_word(msg, "no such column:") or "that column"
            return f"The database doesn't have a column named {col}."
        if "syntax error" in low:
            return "The query had a SQL syntax error and couldn't run."
        if "interrupted" in low or "timeout" in low:
            return "The query took too long and was cancelled."
        return f"The database returned an error: {msg}"

    # Vendor SDK / network errors (pattern-matched by class name to avoid imports)
    if name in ("NotFoundError",):
        return (
            "The LLM returned 404 — usually the configured model name doesn't exist "
            "for this provider. Check the model field in your config."
        )
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return "The LLM rejected the API key. Check your provider credentials."
    if name in ("APIConnectionError", "ConnectError", "ConnectTimeout"):
        return (
            "Couldn't reach the LLM endpoint. If you're using a local server, "
            "make sure it's running."
        )
    if name in ("RateLimitError",):
        return "The LLM is rate-limited right now. Wait a moment and try again."
    if name in ("APITimeoutError", "ReadTimeout"):
        return "The LLM took too long to respond. Try a shorter question or try again."

    # Generic fallback ---------------------------------------------------------
    short = msg or name
    return f"Something went wrong ({name}): {short}"


def _extract_quoted_or_word(msg: str, marker: str) -> str | None:
    """Pull the identifier after a marker like 'no such table:'.

    Returns the identifier surrounded by backticks for readability, or None
    if nothing usable was found.
    """
    lower_msg = msg.lower()
    idx = lower_msg.find(marker.lower())
    if idx < 0:
        return None
    rest = msg[idx + len(marker):].strip()
    if not rest:
        return None
    token = rest.split()[0].strip("\"'`,.;")
    return f"`{token}`" if token else None
