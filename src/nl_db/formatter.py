from __future__ import annotations

import json
from typing import Any

from rich.table import Table

from .executor import QueryResult


def format_as_table(result: QueryResult) -> Table:
    """Build a rich Table for terminal rendering."""
    table = Table(show_header=True, header_style="bold cyan")
    for col in result.columns:
        table.add_column(col)
    for row in result.rows:
        table.add_row(*(_cell(v) for v in row))
    if result.truncated:
        table.caption = f"[yellow]results truncated at {result.row_count} rows[/yellow]"
    return table


def format_as_json(result: QueryResult, *, indent: int | None = 2) -> str:
    """Serialize as JSON: {columns, rows, truncated, row_count}."""
    payload: dict[str, Any] = {
        "columns": list(result.columns),
        "rows": [list(row) for row in result.rows],
        "row_count": result.row_count,
        "truncated": result.truncated,
    }
    return json.dumps(payload, indent=indent, default=_json_default)


def _cell(value: Any) -> str:
    if value is None:
        return "[dim]NULL[/dim]"
    return str(value)


def _json_default(value: Any) -> Any:
    # SQLite returns bytes for BLOB; coerce to repr for JSON.
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
