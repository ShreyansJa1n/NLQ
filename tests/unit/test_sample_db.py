from __future__ import annotations

import sqlite3
from pathlib import Path

from nl_db.schema.sqlite import SQLiteSchemaExtractor


def test_sample_db_builds_and_has_expected_shape(sample_db_path: Path) -> None:
    assert sample_db_path.exists()
    schema = SQLiteSchemaExtractor.from_path(sample_db_path).extract()
    names = schema.table_names()
    assert set(names) == {"users", "categories", "vendors", "transactions"}

    # FKs wired correctly
    txns = schema.table("transactions")
    assert txns is not None
    fk_targets = {(fk.column, fk.references_table) for fk in txns.foreign_keys}
    assert ("user_id", "users") in fk_targets
    assert ("vendor_id", "vendors") in fk_targets
    assert ("category_id", "categories") in fk_targets


def test_sample_db_has_expected_row_counts(sample_db_path: Path) -> None:
    conn = sqlite3.connect(str(sample_db_path))
    try:
        counts = {
            name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in ("users", "categories", "vendors", "transactions")
        }
    finally:
        conn.close()
    assert counts == {"users": 3, "categories": 8, "vendors": 12, "transactions": 31}


def test_sample_db_has_null_categories_and_negative_amounts(
    sample_db_path: Path,
) -> None:
    conn = sqlite3.connect(str(sample_db_path))
    try:
        null_cats = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE category_id IS NULL"
        ).fetchone()[0]
        refunds = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE amount_cents < 0"
        ).fetchone()[0]
    finally:
        conn.close()
    assert null_cats >= 1, "eval queries need NULL category rows"
    assert refunds >= 1, "eval queries need a negative-amount refund"
