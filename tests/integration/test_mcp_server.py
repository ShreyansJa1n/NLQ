from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from nl_db.mcp.server import build_server


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.db"
    conn = sqlite3.connect(str(p))
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            author_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            FOREIGN KEY (author_id) REFERENCES users(id)
        );
        INSERT INTO users VALUES (1, 'alice'), (2, 'bob');
        INSERT INTO posts VALUES (1, 1, 'hello'), (2, 1, 'world'), (3, 2, 'hi');
        """
    )
    conn.commit()
    conn.close()
    return p


@pytest.fixture
def patch_provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace registry.build_provider so build_server uses a canned LLM."""
    from nl_db.llm.provider import ChatResult, Message
    from nl_db.mcp import server as srv

    state: dict[str, Any] = {"queue": []}

    class _Canned:
        name = "canned"
        model = "canned-1"

        def chat(
            self,
            messages: list[Message],
            *,
            temperature: float = 0.0,
            max_output_tokens: int = 1024,
        ) -> ChatResult:
            text = state["queue"].pop(0) if state["queue"] else ""
            return ChatResult(text=text)

    monkeypatch.setattr(srv, "build_provider", lambda _settings: _Canned())

    def queue(*responses: str) -> None:
        state["queue"] = list(responses)

    return queue


def _run_tool(server: Any, name: str, **kwargs: Any) -> Any:
    tool = server._tool_manager.get_tool(name)
    assert tool is not None, f"tool not registered: {name}"
    return asyncio.get_event_loop().run_until_complete(tool.run(arguments=kwargs))


def _read_resource(server: Any, uri: str) -> str:
    contents = asyncio.get_event_loop().run_until_complete(server.read_resource(uri))
    return contents[0].content


def test_list_tables(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db)
    payload = _run_tool(server, "list_tables")
    assert sorted(payload["tables"]) == ["posts", "users"]


def test_describe_schema(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db)
    payload = _run_tool(server, "describe_schema", table_name="posts")
    assert payload["name"] == "posts"
    col_names = [c["name"] for c in payload["columns"]]
    assert col_names == ["id", "author_id", "title"]
    fk = payload["foreign_keys"][0]
    assert fk["references_table"] == "users"


def test_describe_schema_unknown_table_errors(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db)
    with pytest.raises(Exception, match="Table not found"):
        _run_tool(server, "describe_schema", table_name="nope")


def test_query_database_end_to_end(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider(
        "```sql\nSELECT name FROM users ORDER BY id\n```",
        "Returns each user's name.",
    )
    server = build_server(tiny_db)
    payload = _run_tool(server, "query_database", question="list users")
    assert payload["columns"] == ["name"]
    assert payload["rows"] == [["alice"], ["bob"]]
    assert payload["paraphrase"] == "Returns each user's name."
    assert payload["auto_limit_applied"] is True


def test_run_sql_read_only_refuses_destructive(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db, allow_writes=False, expose_run_sql=True)
    with pytest.raises(Exception, match="(?i)destructive"):
        _run_tool(server, "run_sql", sql="DELETE FROM users")


def test_run_sql_with_allow_writes(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db, allow_writes=True, expose_run_sql=True)
    payload = _run_tool(server, "run_sql", sql="DELETE FROM posts WHERE id = 99")
    assert payload["is_destructive"] is True
    assert payload["row_count"] == 0


def test_run_sql_not_registered_by_default(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db)
    assert server._tool_manager.get_tool("run_sql") is None


def test_run_sql_registered_when_exposed(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db, expose_run_sql=True)
    assert server._tool_manager.get_tool("run_sql") is not None


def test_describe_database_tool(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db)
    payload = _run_tool(server, "describe_database")
    assert payload["dialect"] == "sqlite"
    table_names = [t["name"] for t in payload["tables"]]
    assert sorted(table_names) == ["posts", "users"]
    posts = next(t for t in payload["tables"] if t["name"] == "posts")
    assert [c["name"] for c in posts["columns"]] == ["id", "author_id", "title"]
    assert posts["foreign_keys"][0]["references_table"] == "users"


def test_full_schema_resource(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db)
    content = _read_resource(server, "db://schema")
    payload = json.loads(content)
    assert payload["dialect"] == "sqlite"
    table_names = [t["name"] for t in payload["tables"]]
    assert sorted(table_names) == ["posts", "users"]


def test_query_database_returns_cannot_answer(
    tiny_db: Path, patch_provider: Any
) -> None:
    patch_provider("CANNOT_ANSWER: This database has no information about employees.")
    server = build_server(tiny_db)
    payload = _run_tool(server, "query_database", question="list all employees")
    assert payload["state"] == "CANNOT_ANSWER"
    assert "employees" in payload["reason"].lower()
    assert sorted(payload["available_tables"]) == ["posts", "users"]


def test_query_database_returns_clarify(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider("CLARIFY: Do you mean posts by user_id or by title?")
    server = build_server(tiny_db)
    payload = _run_tool(server, "query_database", question="show me posts")
    assert payload["state"] == "CLARIFY"
    assert "user_id or by title" in payload["question"]


def test_query_database_returns_answer_state(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider(
        "```sql\nSELECT title FROM posts ORDER BY id\n```",
        "Lists every post title.",
    )
    server = build_server(tiny_db)
    payload = _run_tool(server, "query_database", question="list titles")
    assert payload["state"] == "ANSWER"
    assert payload["columns"] == ["title"]
    assert payload["rows"] == [["hello"], ["world"], ["hi"]]


def test_schema_resource(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    server = build_server(tiny_db)
    content = _read_resource(server, "db://schema/users")
    payload = json.loads(content)
    assert payload["name"] == "users"
    col_names = [c["name"] for c in payload["columns"]]
    assert col_names == ["id", "name"]


def test_tool_annotations_set_correctly(tiny_db: Path, patch_provider: Any) -> None:
    patch_provider()
    ro_server = build_server(tiny_db, allow_writes=False, expose_run_sql=True)
    wr_server = build_server(tiny_db, allow_writes=True, expose_run_sql=True)

    ro_run_sql = ro_server._tool_manager.get_tool("run_sql")
    wr_run_sql = wr_server._tool_manager.get_tool("run_sql")

    assert ro_run_sql.annotations.readOnlyHint is True
    assert ro_run_sql.annotations.destructiveHint is False
    assert wr_run_sql.annotations.readOnlyHint is False
    assert wr_run_sql.annotations.destructiveHint is True

    list_tool = ro_server._tool_manager.get_tool("list_tables")
    assert list_tool.annotations.readOnlyHint is True
    assert list_tool.annotations.destructiveHint is False


def test_main_rejects_allow_writes_without_expose_run_sql(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from nl_db.mcp.server import main

    db = tmp_path / "x.db"
    db.touch()
    exit_code = main(["--db", str(db), "--allow-writes"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "requires --expose-run-sql" in captured.out
