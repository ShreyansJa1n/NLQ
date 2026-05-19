from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .base import Schema


class SchemaCache:
    """In-memory schema cache keyed by (db_path, mtime).

    Invalidates automatically when the underlying file changes.
    """

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[float, Schema]] = {}

    def get(
        self, db_path: Path, extract: Callable[[], Schema]
    ) -> Schema:
        mtime = db_path.stat().st_mtime
        cached = self._entries.get(db_path)
        if cached and cached[0] == mtime:
            return cached[1]
        schema = extract()
        self._entries[db_path] = (mtime, schema)
        return schema

    def invalidate(self, db_path: Path | None = None) -> None:
        if db_path is None:
            self._entries.clear()
        else:
            self._entries.pop(db_path, None)
