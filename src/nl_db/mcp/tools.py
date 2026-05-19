"""Tool descriptions for the MCP server.

These strings are PRODUCT COPY — host LLMs read them to decide when and how
to call each tool. They are written for the host model, not the human reader.

Keep them:
- Concrete: name actual capabilities, not abstract goals.
- Bounded: state what the tool will NOT do, so the host doesn't try.
- Examples-first when behavior is non-obvious.
"""
from __future__ import annotations

LIST_TABLES_DESC = """\
List every user-created table in the connected database.

Returns a JSON object: {"tables": ["users", "transactions", ...]}.
No arguments. Cheap to call. For most use cases prefer describe_database()
which returns table names AND columns in one call.

System tables (sqlite_*) are excluded.
"""

DESCRIBE_DATABASE_DESC = """\
Return the full schema for the entire database in one call: every table
with its columns (name, type, nullability, primary keys) and foreign-key
relationships.

No arguments.

Returns a JSON object: {
  "dialect": "sqlite",
  "tables": [
    {
      "name": "users",
      "columns": [{"name": "id", "type": "INTEGER", "nullable": false, "primary_key": true, "default": null}, ...],
      "foreign_keys": [{"column": "user_id", "references_table": "users", "references_column": "id"}, ...]
    },
    ...
  ]
}

**Call this once at the start of a conversation** so you know what the
database actually contains before asking natural-language questions. It's
strictly more informative than list_tables and costs the same single round-trip.

The same payload is also exposed as the Resource `db://schema` — read it
that way if your client prefers Resources to tool calls.
"""

DESCRIBE_SCHEMA_DESC = """\
Return the schema for ONE specific table. Useful only if you already know the
table name and want to focus on its columns/FKs without pulling the full
database schema.

For most use cases describe_database() (one call, everything) is the better
starting point.

Argument:
- table_name (string): exact name from list_tables or describe_database.

Will return an error if table_name does not exist.
"""

QUERY_DATABASE_DESC = """\
Answer a plain-English question about the database.

You should NOT write SQL yourself — nl-db generates, validates, and runs the
SQL behind the scenes. Just send the user's question in plain English.

**Before your first call in a conversation, call `describe_database()` (or
read the `db://schema` resource) so you know what tables and columns are
available.** nl-db will refuse questions that don't match the schema, so
grounding upfront saves round-trips.

Argument:
- question (string): a plain-English question, e.g.
  "How much did Alice spend on groceries last month?"

Returns ONE of three response shapes, distinguished by the "state" field:

  ANSWER (the question was answerable):
    {
      "state": "ANSWER",
      "sql": "<the SQL that ran>",
      "paraphrase": "<plain-English explanation of the SQL>",
      "columns": [...],
      "rows": [...],
      "row_count": N,
      "truncated": bool,
      "auto_limit_applied": bool
    }

  CANNOT_ANSWER (the schema doesn't contain the data needed):
    {
      "state": "CANNOT_ANSWER",
      "reason": "<plain-English reason>",
      "available_tables": ["users", "transactions", ...]
    }
    Use this signal to tell the user what the database actually tracks.

  CLARIFY (the question is ambiguous):
    {
      "state": "CLARIFY",
      "question": "<plain-English follow-up question>"
    }
    Ask the user this question, then call query_database again with the
    clarified intent.

This tool is read-only. It refuses INSERT/UPDATE/DELETE and auto-injects
LIMIT on unbounded SELECTs.
"""

RUN_SQL_DESC = """\
Execute a SQL statement the caller has already written.

Argument:
- sql (string): a complete SQL statement.

Returns: {sql, columns, rows, row_count, truncated, is_destructive}.
This tool is **only registered when the server was started with `--expose-run-sql`**.
Prefer query_database() — sending NL is the canonical path, and nl-db handles
the SQL translation for you.

DESTRUCTIVE: if --allow-writes was also passed at server start, this tool
will run INSERT/UPDATE/DELETE/DROP/ALTER. Otherwise it refuses writes.
Auto-LIMIT is NOT injected here — include your own LIMIT if you want one.
"""

SCHEMA_RESOURCE_DESC = """\
Schema for a single table, exposed as a Resource so host models can browse
without calling a tool. URI form: db://schema/<table_name>.

For the full database schema in one resource, read db://schema (no table name).
"""

FULL_SCHEMA_RESOURCE_DESC = """\
The full database schema — every table with its columns, primary keys, and
foreign keys — as a single JSON document.

URI: db://schema. Read this once at the start of a conversation to ground
yourself in what the database contains before asking NL questions via
query_database. Same payload as the describe_database() tool.
"""
