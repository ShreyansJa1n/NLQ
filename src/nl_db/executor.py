from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class QueryResult:
    columns: tuple[str, ...]
    rows: list[tuple[Any, ...]]
    truncated: bool = False
    row_count: int = 0


class QueryExecutor(Protocol):
    def execute(self, sql: str) -> QueryResult: ...


class SQLiteExecutor:
    """Executes a SQL statement against a SQLite DB with timeout + row cap.

    Timeout uses sqlite3's `Connection(timeout=...)` for lock waits plus a
    `progress_handler` interrupt for runaway queries.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        timeout_s: float = 10.0,
        max_rows: int | None = 1000,
    ) -> None:
        self._db_path = db_path
        self._timeout_s = timeout_s
        self._max_rows = max_rows

    def execute(self, sql: str) -> QueryResult:
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=self._timeout_s,
            isolation_level=None,  # autocommit; pure-read default
        )
        # Approx 1M VM ops between progress checks; aborts runaway queries.
        # SQLite is fast enough that 1M ops ~= a few ms on typical hardware.
        deadline_op_budget = 10_000_000
        ops_seen = [0]

        def _progress() -> int:
            ops_seen[0] += 1
            return 1 if ops_seen[0] > deadline_op_budget else 0

        conn.set_progress_handler(_progress, 1_000_000)
        try:
            cur = conn.execute(sql)
            description = cur.description or ()
            columns = tuple(d[0] for d in description)
            cap = self._max_rows
            rows: list[tuple[Any, ...]] = []
            truncated = False
            for row in cur:
                if cap is not None and len(rows) >= cap:
                    truncated = True
                    break
                rows.append(tuple(row))
            return QueryResult(
                columns=columns,
                rows=rows,
                truncated=truncated,
                row_count=len(rows),
            )
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()
