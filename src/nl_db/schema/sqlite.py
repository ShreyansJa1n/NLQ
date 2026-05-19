from __future__ import annotations

import sqlite3
from pathlib import Path

from .base import Column, ForeignKey, Schema, Table


class SQLiteSchemaExtractor:
    """Extracts a Schema from a SQLite database via PRAGMA queries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection

    @classmethod
    def from_path(cls, path: Path | str) -> SQLiteSchemaExtractor:
        conn = sqlite3.connect(str(path))
        return cls(conn)

    def extract(self) -> Schema:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        table_names = [row[0] for row in cur.fetchall()]
        tables = tuple(self._extract_table(name) for name in table_names)
        return Schema(dialect="sqlite", tables=tables)

    def _extract_table(self, name: str) -> Table:
        cur = self._conn.cursor()
        cur.execute(f'PRAGMA table_info("{name}")')
        columns = tuple(
            Column(
                name=row[1],
                type=(row[2] or "").upper() or "ANY",
                nullable=(row[3] == 0),
                primary_key=bool(row[5]),
                default=str(row[4]) if row[4] is not None else None,
            )
            for row in cur.fetchall()
        )

        cur.execute(f'PRAGMA foreign_key_list("{name}")')
        foreign_keys = tuple(
            ForeignKey(
                column=row[3],
                references_table=row[2],
                references_column=row[4],
            )
            for row in cur.fetchall()
        )
        return Table(name=name, columns=columns, foreign_keys=foreign_keys)
