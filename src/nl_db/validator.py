from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


class SQLValidationError(ValueError):
    pass


@dataclass
class ValidationResult:
    sql: str  # may differ from input (e.g. auto-LIMIT injected)
    auto_limit_applied: bool
    is_destructive: bool


_DESTRUCTIVE_KINDS: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
)


def validate_sql(
    sql: str,
    *,
    dialect: str = "sqlite",
    allow_writes: bool = False,
    max_rows: int | None = 1000,
) -> ValidationResult:
    """Parse, classify, and (for safe SELECTs) auto-LIMIT a SQL statement.

    Rules:
    - Parsing failure → SQLValidationError.
    - More than one top-level statement → SQLValidationError.
    - Destructive statement (INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE)
      requires allow_writes=True.
    - If max_rows is set and the statement is a bare SELECT without LIMIT,
      a LIMIT clause is injected.
    """
    try:
        parsed = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        raise SQLValidationError(f"SQL parse error: {e}") from e

    statements = [stmt for stmt in parsed if stmt is not None]
    if not statements:
        raise SQLValidationError("No SQL statement found")
    if len(statements) > 1:
        raise SQLValidationError(
            f"Expected exactly one statement, got {len(statements)}"
        )

    statement = statements[0]
    is_destructive = isinstance(statement, _DESTRUCTIVE_KINDS) or any(
        isinstance(node, _DESTRUCTIVE_KINDS) for node in statement.walk()
    )

    if is_destructive and not allow_writes:
        raise SQLValidationError(
            "Refusing to run destructive SQL (INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE). "
            "Pass --allow-writes to override."
        )

    auto_limit_applied = False
    final_statement = statement
    if (
        not is_destructive
        and max_rows is not None
        and isinstance(statement, exp.Select)
        and statement.args.get("limit") is None
    ):
        final_statement = statement.limit(max_rows, copy=True)
        auto_limit_applied = True

    return ValidationResult(
        sql=final_statement.sql(dialect=dialect),
        auto_limit_applied=auto_limit_applied,
        is_destructive=is_destructive,
    )
