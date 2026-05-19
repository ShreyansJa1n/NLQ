"""MCP stdio server exposing the nl-db pipeline.

Tools:
- list_tables          → table names
- describe_database    → full schema (one call, everything)
- describe_schema      → schema for one specific table
- query_database       → NL → outcome (Answer / CannotAnswer / Clarify)
- run_sql              → execute raw SQL (only when --expose-run-sql is set)

Resources:
- db://schema          → full schema, same payload as describe_database
- db://schema/<table>  → schema for one table, same as describe_schema

Run:
    uv run nl-db-mcp --db path/to.db
    uv run nl-db-mcp --db path/to.db --expose-run-sql              # opt in to raw-SQL tool
    uv run nl-db-mcp --db path/to.db --expose-run-sql --allow-writes
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ..config import load_settings
from ..conversation import Conversation, Turn, summarize_rows
from ..generator import Answer, CannotAnswer, Clarify
from ..llm.registry import build_provider
from ..pipeline import Pipeline
from ..schema.base import Schema
from ..validator import SQLValidationError, validate_sql
from . import tools as descriptions


def _schema_to_dict(schema: Schema) -> dict[str, Any]:
    return {
        "dialect": schema.dialect,
        "tables": [
            {
                "name": t.name,
                "columns": [
                    {
                        "name": c.name,
                        "type": c.type,
                        "nullable": c.nullable,
                        "primary_key": c.primary_key,
                        "default": c.default,
                    }
                    for c in t.columns
                ],
                "foreign_keys": [
                    {
                        "column": fk.column,
                        "references_table": fk.references_table,
                        "references_column": fk.references_column,
                    }
                    for fk in t.foreign_keys
                ],
            }
            for t in schema.tables
        ],
    }


def _table_to_dict(schema: Schema, table_name: str) -> dict[str, Any]:
    t = schema.table(table_name)
    if t is None:
        raise ValueError(f"Table not found: {table_name!r}. Use list_tables.")
    return {
        "name": t.name,
        "columns": [
            {
                "name": c.name,
                "type": c.type,
                "nullable": c.nullable,
                "primary_key": c.primary_key,
                "default": c.default,
            }
            for c in t.columns
        ],
        "foreign_keys": [
            {
                "column": fk.column,
                "references_table": fk.references_table,
                "references_column": fk.references_column,
            }
            for fk in t.foreign_keys
        ],
    }


def build_server(
    db_path: Path,
    *,
    allow_writes: bool = False,
    expose_run_sql: bool = False,
) -> FastMCP:
    """Construct a FastMCP server bound to the given DB path.

    `run_sql` is registered ONLY when `expose_run_sql=True`. `allow_writes`
    is meaningful only in combination with `expose_run_sql` — writes are
    reachable only through `run_sql`.
    """
    settings = load_settings()
    settings.db.path = db_path
    provider = build_provider(settings)
    pipeline = Pipeline(
        provider=provider,
        db_path=db_path,
        max_rows=settings.limits.max_rows,
        timeout_s=settings.limits.timeout_s,
        paraphrase=True,
    )

    mcp = FastMCP(
        name="nl-db",
        instructions=(
            "Natural-language gateway to a SQLite database. Call "
            "describe_database() once at the start to ground yourself in the "
            "schema, then send NL questions via query_database(). nl-db "
            "writes and runs the SQL for you — don't translate to SQL yourself."
        ),
    )

    @mcp.tool(
        name="list_tables",
        description=descriptions.LIST_TABLES_DESC,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    def list_tables() -> dict[str, list[str]]:
        return {"tables": list(pipeline.schema().table_names())}

    @mcp.tool(
        name="describe_database",
        description=descriptions.DESCRIBE_DATABASE_DESC,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    def describe_database() -> dict[str, Any]:
        return _schema_to_dict(pipeline.schema())

    @mcp.tool(
        name="describe_schema",
        description=descriptions.DESCRIBE_SCHEMA_DESC,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    def describe_schema(table_name: str) -> dict[str, Any]:
        return _table_to_dict(pipeline.schema(), table_name)

    # In-memory conversation store. Keyed by conversation_id; entries last for
    # the lifetime of this server process. Host LLMs that want multi-turn
    # should generate a UUID at conversation start and reuse it for follow-ups.
    conversations: dict[str, Conversation] = {}

    @mcp.tool(
        name="query_database",
        description=descriptions.QUERY_DATABASE_DESC,
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
    )
    def query_database(
        question: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        history = None
        if conversation_id is not None:
            history = conversations.setdefault(conversation_id, Conversation())

        output = pipeline.run(question, allow_writes=False, history=history)
        outcome = output.outcome

        # Record the turn into history if a conversation_id was supplied.
        if history is not None:
            row_summary: str | None = None
            if isinstance(outcome, Answer) and output.result is not None:
                row_summary = summarize_rows(
                    output.result.columns, output.result.rows
                )
            history.append(
                Turn(question=question, outcome=outcome, row_summary=row_summary)
            )

        if isinstance(outcome, CannotAnswer):
            return {
                "state": "CANNOT_ANSWER",
                "reason": outcome.reason,
                "available_tables": list(outcome.available_tables),
            }
        if isinstance(outcome, Clarify):
            return {
                "state": "CLARIFY",
                "question": outcome.question,
            }
        # Answer branch
        assert isinstance(outcome, Answer)
        assert output.result is not None
        return {
            "state": "ANSWER",
            "sql": output.sql_final,
            "paraphrase": output.paraphrase,
            "columns": list(output.result.columns),
            "rows": [list(row) for row in output.result.rows],
            "row_count": output.result.row_count,
            "truncated": output.result.truncated,
            "auto_limit_applied": output.auto_limit_applied,
        }

    if expose_run_sql:
        @mcp.tool(
            name="run_sql",
            description=descriptions.RUN_SQL_DESC,
            annotations=ToolAnnotations(
                readOnlyHint=not allow_writes,
                destructiveHint=allow_writes,
            ),
        )
        def run_sql(sql: str) -> dict[str, Any]:
            validation = validate_sql(
                sql,
                dialect=pipeline.schema().dialect,
                allow_writes=allow_writes,
                max_rows=None,
            )
            result = pipeline._executor.execute(validation.sql)  # noqa: SLF001
            return {
                "sql": validation.sql,
                "is_destructive": validation.is_destructive,
                "columns": list(result.columns),
                "rows": [list(row) for row in result.rows],
                "row_count": result.row_count,
                "truncated": result.truncated,
            }

    @mcp.resource(
        "db://schema",
        description=descriptions.FULL_SCHEMA_RESOURCE_DESC,
        mime_type="application/json",
    )
    def full_schema_resource() -> str:
        return json.dumps(_schema_to_dict(pipeline.schema()), indent=2)

    @mcp.resource(
        "db://schema/{table_name}",
        description=descriptions.SCHEMA_RESOURCE_DESC,
        mime_type="application/json",
    )
    def schema_resource(table_name: str) -> str:
        return json.dumps(_table_to_dict(pipeline.schema(), table_name), indent=2)

    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="nl-db MCP stdio server",
    )
    parser.add_argument("--db", type=Path, required=True, help="SQLite database path.")
    parser.add_argument(
        "--expose-run-sql",
        action="store_true",
        help=(
            "Register the run_sql tool. Off by default — host LLMs should "
            "use query_database (NL) instead of writing SQL themselves."
        ),
    )
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help=(
            "Permit INSERT/UPDATE/DELETE/DROP/ALTER via run_sql. Requires "
            "--expose-run-sql since writes are only reachable through that tool."
        ),
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"ERROR: database not found: {args.db}", flush=True)
        return 1

    if args.allow_writes and not args.expose_run_sql:
        print(
            "ERROR: --allow-writes requires --expose-run-sql "
            "(writes are reachable only through the run_sql tool).",
            flush=True,
        )
        return 2

    server = build_server(
        args.db,
        allow_writes=args.allow_writes,
        expose_run_sql=args.expose_run_sql,
    )
    # FastMCP.run uses stdio by default; blocks until client disconnects.
    server.run(transport="stdio")
    return 0


# Wrap the run_sql validation error so it surfaces cleanly through MCP.
# (FastMCP catches exceptions and returns them as tool errors, so just let
# SQLValidationError propagate.)
__all__ = ["build_server", "main", "SQLValidationError"]


if __name__ == "__main__":
    # Suppress FastMCP's stderr banner during quiet test runs.
    if os.environ.get("NL_DB_MCP_QUIET"):
        import sys as _sys
        _sys.stderr = open(os.devnull, "w")  # noqa: SIM115
    raise SystemExit(main())
