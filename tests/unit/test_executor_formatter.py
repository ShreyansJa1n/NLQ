from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from rich.console import Console

from nl_db.executor import SQLiteExecutor
from nl_db.formatter import format_as_json, format_as_table


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.db"
    conn = sqlite3.connect(str(p))
    conn.executescript(
        """
        CREATE TABLE t (id INTEGER PRIMARY KEY, label TEXT, score REAL);
        INSERT INTO t VALUES (1, 'a', 1.5), (2, 'b', NULL), (3, 'c', 3.14);
        """
    )
    conn.commit()
    conn.close()
    return p


def test_executor_returns_columns_and_rows(tiny_db: Path) -> None:
    result = SQLiteExecutor(tiny_db).execute("SELECT id, label, score FROM t ORDER BY id;")
    assert result.columns == ("id", "label", "score")
    assert result.row_count == 3
    assert result.rows[0] == (1, "a", 1.5)
    assert result.rows[1] == (2, "b", None)
    assert result.truncated is False


def test_executor_truncates_at_row_cap(tiny_db: Path) -> None:
    result = SQLiteExecutor(tiny_db, max_rows=2).execute("SELECT * FROM t ORDER BY id;")
    assert result.row_count == 2
    assert result.truncated is True


def test_format_as_table_includes_columns_and_rows(tiny_db: Path) -> None:
    result = SQLiteExecutor(tiny_db).execute("SELECT id, label FROM t ORDER BY id;")
    table = format_as_table(result)
    # Render to plain text via Console capture
    console = Console(record=True, width=80)
    console.print(table)
    rendered = console.export_text()
    assert "id" in rendered
    assert "label" in rendered
    assert "a" in rendered and "b" in rendered and "c" in rendered


def test_format_as_table_marks_truncation(tiny_db: Path) -> None:
    result = SQLiteExecutor(tiny_db, max_rows=1).execute("SELECT * FROM t ORDER BY id;")
    table = format_as_table(result)
    assert table.caption is not None
    assert "truncated" in str(table.caption)


def test_format_as_json_shape(tiny_db: Path) -> None:
    result = SQLiteExecutor(tiny_db).execute("SELECT id, label FROM t ORDER BY id;")
    payload = json.loads(format_as_json(result))
    assert payload["columns"] == ["id", "label"]
    assert payload["rows"] == [[1, "a"], [2, "b"], [3, "c"]]
    assert payload["row_count"] == 3
    assert payload["truncated"] is False


def test_format_as_json_handles_null(tiny_db: Path) -> None:
    result = SQLiteExecutor(tiny_db).execute("SELECT score FROM t WHERE id = 2;")
    payload = json.loads(format_as_json(result))
    assert payload["rows"] == [[None]]
