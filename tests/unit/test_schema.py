from __future__ import annotations

import sqlite3
from pathlib import Path

from nl_db.schema.base import render_for_prompt
from nl_db.schema.cache import SchemaCache
from nl_db.schema.sqlite import SQLiteSchemaExtractor


def _make_sample_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL
        );
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            category_id INTEGER,
            amount_cents INTEGER NOT NULL,
            occurred_on TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (category_id) REFERENCES categories(id)
        );
        """
    )
    conn.commit()
    return conn


def test_sqlite_extractor_extracts_tables_columns_pks() -> None:
    schema = SQLiteSchemaExtractor(_make_sample_conn()).extract()

    assert schema.dialect == "sqlite"
    assert schema.table_names() == ("categories", "transactions", "users")

    users = schema.table("users")
    assert users is not None
    assert [c.name for c in users.columns] == ["id", "email", "name", "created_at"]
    id_col = users.columns[0]
    assert id_col.primary_key is True
    assert id_col.type == "INTEGER"

    email_col = users.columns[1]
    assert email_col.nullable is False
    assert email_col.primary_key is False


def test_sqlite_extractor_extracts_foreign_keys() -> None:
    schema = SQLiteSchemaExtractor(_make_sample_conn()).extract()
    txns = schema.table("transactions")
    assert txns is not None
    fk_map = {fk.column: (fk.references_table, fk.references_column) for fk in txns.foreign_keys}
    assert fk_map["user_id"] == ("users", "id")
    assert fk_map["category_id"] == ("categories", "id")


def test_render_for_prompt_is_compact_and_includes_fk_arrows() -> None:
    schema = SQLiteSchemaExtractor(_make_sample_conn()).extract()
    rendered = render_for_prompt(schema)

    assert "Table users:" in rendered
    assert "Table transactions:" in rendered
    assert "PRIMARY KEY" in rendered
    assert "-> users.id" in rendered
    assert "-> categories.id" in rendered
    # token-efficient: no SQL keywords like CREATE, REFERENCES
    assert "CREATE" not in rendered
    assert "REFERENCES" not in rendered


def test_render_for_prompt_empty_schema() -> None:
    from nl_db.schema.base import Schema

    out = render_for_prompt(Schema(dialect="sqlite", tables=()))
    assert "empty sqlite" in out


def test_schema_cache_hits_when_mtime_unchanged(tmp_path: Path) -> None:
    db_path = tmp_path / "x.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY);")
    conn.commit()
    conn.close()

    cache = SchemaCache()
    calls = {"n": 0}

    def extract() -> object:
        calls["n"] += 1
        return SQLiteSchemaExtractor.from_path(db_path).extract()

    s1 = cache.get(db_path, extract)  # type: ignore[arg-type]
    s2 = cache.get(db_path, extract)  # type: ignore[arg-type]
    assert s1 is s2
    assert calls["n"] == 1


def test_schema_cache_invalidates_when_mtime_changes(tmp_path: Path) -> None:
    db_path = tmp_path / "x.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY);")
    conn.commit()
    conn.close()

    cache = SchemaCache()
    calls = {"n": 0}

    def extract() -> object:
        calls["n"] += 1
        return SQLiteSchemaExtractor.from_path(db_path).extract()

    cache.get(db_path, extract)  # type: ignore[arg-type]

    # mutate file to bump mtime
    import os
    import time

    time.sleep(0.01)
    new_mtime = db_path.stat().st_mtime + 1
    os.utime(db_path, (new_mtime, new_mtime))

    cache.get(db_path, extract)  # type: ignore[arg-type]
    assert calls["n"] == 2
