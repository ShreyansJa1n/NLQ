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
No arguments. Cheap to call — use it before any other tool whenever you're
unsure what data is available.

System tables (sqlite_*) are excluded.
"""

DESCRIBE_SCHEMA_DESC = """\
Return the full schema for one table: columns with types and nullability,
primary keys, and any foreign-key references.

Argument:
- table_name (string): exact name from list_tables. Case-sensitive.

Returns a JSON object describing the table. Use this to ground SQL
generation in the actual column names — never guess columns.

Will return an error if table_name does not exist.
"""

QUERY_DATABASE_DESC = """\
Translate a natural-language question into SQL, validate it, and run it.

Argument:
- question (string): a plain-English question, e.g.
  "How much did Alice spend on groceries last month?"

Returns a JSON object: {
  "sql": "<the SQL that ran>",
  "paraphrase": "<plain-English explanation of the SQL>",
  "columns": [...],
  "rows": [...],
  "row_count": N,
  "truncated": bool,
  "auto_limit_applied": bool
}

This tool is READ-ONLY: it refuses INSERT/UPDATE/DELETE/DROP/ALTER and
auto-injects LIMIT on unbounded SELECTs. For writes, use run_sql with
allow_writes enabled at server start.

Use this whenever the user asks a question the database can answer.
"""

RUN_SQL_DESC = """\
Execute a SQL statement the caller has already written.

Argument:
- sql (string): a complete SQL statement (single statement, no semicolon
  required).

Returns the same shape as query_database: {sql, columns, rows, ...}.
Use this when you (the host model) already know the exact SQL you want
to run — typically after inspecting the schema. Prefer query_database
when starting from a natural-language question.

DESTRUCTIVE: if --allow-writes was passed at server start, this tool
will run INSERT/UPDATE/DELETE/DROP/ALTER. Otherwise it behaves like
query_database and refuses writes. Auto-LIMIT is NOT injected here —
include your own LIMIT if you want one.
"""

SCHEMA_RESOURCE_DESC = """\
Schema for a single table, exposed as a Resource so host models can browse
without calling a tool. URI form: db://schema/<table_name>.
"""
