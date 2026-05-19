"""Deterministically build the sample personal-finance SQLite database.

Schema is small but non-trivial — enough to exercise filtering, joins,
aggregation, dates, NULLs, and FK navigation in the eval harness.

Run directly: `uv run python tests/fixtures/build_sample_db.py`
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE_DB = HERE / "sample.db"

SCHEMA = """
CREATE TABLE users (
    id          INTEGER PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE categories (
    id          INTEGER PRIMARY KEY,
    label       TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL CHECK (kind IN ('expense', 'income'))
);

CREATE TABLE vendors (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE TABLE transactions (
    id            INTEGER PRIMARY KEY,
    user_id       INTEGER NOT NULL,
    vendor_id     INTEGER,
    category_id   INTEGER,
    amount_cents  INTEGER NOT NULL,
    occurred_on   TEXT NOT NULL,
    note          TEXT,
    FOREIGN KEY (user_id)     REFERENCES users(id),
    FOREIGN KEY (vendor_id)   REFERENCES vendors(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

CREATE INDEX idx_txn_user_date ON transactions (user_id, occurred_on);
"""

USERS = [
    (1, "alice@example.com", "Alice",   "2025-08-15"),
    (2, "bob@example.com",   "Bob",     "2025-11-01"),
    (3, "carol@example.com", "Carol",   "2026-01-20"),
]

CATEGORIES = [
    (1, "groceries",     "expense"),
    (2, "rent",          "expense"),
    (3, "dining",        "expense"),
    (4, "transport",     "expense"),
    (5, "utilities",     "expense"),
    (6, "entertainment", "expense"),
    (7, "salary",        "income"),
    (8, "refund",        "income"),
]

VENDORS = [
    (1, "Whole Foods",      1),
    (2, "Trader Joe's",     1),
    (3, "Landlord LLC",     2),
    (4, "Chipotle",         3),
    (5, "Sweetgreen",       3),
    (6, "Uber",             4),
    (7, "MBTA",             4),
    (8, "Eversource",       5),
    (9, "Comcast",          5),
    (10, "Netflix",         6),
    (11, "Acme Corp",       7),
    (12, "Unknown Vendor",  None),
]

# Anchor the calendar so the data is reproducible regardless of when tests run.
ANCHOR = date(2026, 5, 15)


def _d(days_ago: int) -> str:
    return (ANCHOR - timedelta(days=days_ago)).isoformat()


# (id, user_id, vendor_id, category_id, amount_cents, days_ago, note)
TRANSACTIONS: list[tuple[int, int, int | None, int | None, int, int, str | None]] = [
    # Alice — current month (May 2026)
    (1,  1, 1, 1, 4200,    3, "weekly grocery run"),
    (2,  1, 4, 3, 1850,    4, None),
    (3,  1, 3, 2, 180000,  10, "May rent"),
    (4,  1, 6, 4, 1200,    5, None),
    (5,  1, 10, 6, 1599,   1, "Netflix monthly"),
    (6,  1, 11, 7, 500000, 14, "May salary"),
    # Alice — previous month (April 2026)
    (7,  1, 2, 1, 3700,    35, None),
    (8,  1, 5, 3, 2200,    33, None),
    (9,  1, 3, 2, 180000,  40, "April rent"),
    (10, 1, 8, 5, 8900,    36, "electric bill"),
    (11, 1, 11, 7, 500000, 45, "April salary"),
    (12, 1, 12, None, 2500, 30, "uncategorized cash"),

    # Bob — current month
    (13, 2, 2, 1, 5200,    2, None),
    (14, 2, 5, 3, 1700,    6, None),
    (15, 2, 7, 4, 250,     1, "bus pass"),
    (16, 2, 9, 5, 7500,    9, None),
    (17, 2, 11, 7, 420000, 13, "May payroll"),
    # Bob — previous month
    (18, 2, 1, 1, 4800,    37, None),
    (19, 2, 4, 3, 1500,    34, None),
    (20, 2, 6, 4, 800,     38, None),
    (21, 2, 11, 7, 420000, 44, "April payroll"),

    # Carol — current month
    (22, 3, 1, 1, 6100,    7, None),
    (23, 3, 3, 2, 215000,  11, "May rent"),
    (24, 3, 6, 4, 950,     2, None),
    (25, 3, 10, 6, 1599,   1, "Netflix"),
    (26, 3, 11, 7, 600000, 15, None),
    # A NULL category transaction
    (27, 3, 12, None, 4200, 8, "split-bill cash"),
    # Carol — previous month
    (28, 3, 2, 1, 5500,    33, None),
    (29, 3, 3, 2, 215000,  41, "April rent"),
    (30, 3, 11, 7, 600000, 46, "April salary"),
    # A refund
    (31, 3, 1, 8, -1500,   12, "produce return refund"),
]


def build(db_path: Path = SAMPLE_DB, *, overwrite: bool = True) -> Path:
    if db_path.exists() and overwrite:
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", USERS)
        conn.executemany("INSERT INTO categories VALUES (?, ?, ?)", CATEGORIES)
        conn.executemany("INSERT INTO vendors VALUES (?, ?, ?)", VENDORS)
        conn.executemany(
            "INSERT INTO transactions "
            "(id, user_id, vendor_id, category_id, amount_cents, occurred_on, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (id_, uid, vid, cid, amt, _d(days), note)
                for (id_, uid, vid, cid, amt, days, note) in TRANSACTIONS
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


if __name__ == "__main__":
    out = build()
    print(f"built {out}")
