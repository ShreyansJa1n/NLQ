from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool
    primary_key: bool = False
    default: str | None = None


@dataclass(frozen=True)
class ForeignKey:
    column: str
    references_table: str
    references_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()

    def primary_keys(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.primary_key)


@dataclass(frozen=True)
class Schema:
    dialect: str
    tables: tuple[Table, ...] = field(default_factory=tuple)

    def table(self, name: str) -> Table | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    def table_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tables)


class SchemaExtractor(Protocol):
    """Extracts a Schema from a live database connection."""

    def extract(self) -> Schema: ...


def render_for_prompt(schema: Schema) -> str:
    """Render schema as a compact, token-efficient string for prompt injection.

    Format example:
        Table users:
          id INTEGER PRIMARY KEY
          email TEXT NOT NULL
          name TEXT
        Table transactions:
          id INTEGER PRIMARY KEY
          user_id INTEGER NOT NULL -> users.id
          amount_cents INTEGER NOT NULL
          ...
    """
    if not schema.tables:
        return f"(empty {schema.dialect} database)"

    lines: list[str] = []
    for table in schema.tables:
        lines.append(f"Table {table.name}:")
        fk_by_column = {fk.column: fk for fk in table.foreign_keys}
        for col in table.columns:
            parts = [f"  {col.name} {col.type}"]
            if col.primary_key:
                parts.append("PRIMARY KEY")
            if not col.nullable and not col.primary_key:
                parts.append("NOT NULL")
            if col.default is not None:
                parts.append(f"DEFAULT {col.default}")
            fk = fk_by_column.get(col.name)
            if fk:
                parts.append(f"-> {fk.references_table}.{fk.references_column}")
            lines.append(" ".join(parts))
    return "\n".join(lines)
