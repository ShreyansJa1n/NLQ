from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import nl_db.cli as cli
from nl_db.llm.provider import ChatResult, Message


class CannedProvider:
    name = "canned"
    model = "canned-1"

    def __init__(self, *responses: str) -> None:
        self._queue = list(responses)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> ChatResult:
        text = self._queue.pop(0) if self._queue else ""
        return ChatResult(text=text)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.db"
    conn = sqlite3.connect(str(p))
    conn.executescript(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, label TEXT); "
        "INSERT INTO t VALUES (1, 'a'), (2, 'b');"
    )
    conn.commit()
    conn.close()
    return p


@pytest.fixture
def patch_provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace build_provider so the CLI gets a CannedProvider, not a real LLM."""
    state: dict[str, CannedProvider | None] = {"current": None}

    def set_provider(*responses: str) -> CannedProvider:
        p = CannedProvider(*responses)
        state["current"] = p
        return p

    def fake_build(_settings: Any) -> CannedProvider:
        prov = state["current"]
        assert prov is not None, "test forgot to call patch_provider(...)"
        return prov

    monkeypatch.setattr(cli, "build_provider", fake_build)
    return set_provider


def test_query_command_renders_table(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_provider: Any
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr(
        "nl_db.config.default_config_path", lambda: tmp_path / "missing.toml"
    )
    monkeypatch.setattr("nl_db.cli.load_settings", _load_with_logdir(tmp_path))
    patch_provider(
        "```sql\nSELECT id, label FROM t ORDER BY id\n```",
        "Lists the two rows.",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["query", "list everything", "--db", str(db), "--no-confirm"]
    )
    assert result.exit_code == 0, result.output
    assert "Generated SQL" in result.output
    assert "In plain English" in result.output
    assert "Lists the two rows." in result.output
    assert "a" in result.output and "b" in result.output


def test_query_command_json(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_provider: Any
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr(
        "nl_db.config.default_config_path", lambda: tmp_path / "missing.toml"
    )
    monkeypatch.setattr("nl_db.cli.load_settings", _load_with_logdir(tmp_path))
    patch_provider(
        "```sql\nSELECT id FROM t ORDER BY id\n```",
        "Lists ids.",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["query", "ids", "--db", str(db), "--no-confirm", "--json"]
    )
    assert result.exit_code == 0, result.output
    # parse the JSON block out of mixed stdout
    json_start = result.output.index("{")
    json_end = result.output.rindex("}") + 1
    payload = json.loads(result.output[json_start:json_end])
    assert payload["columns"] == ["id"]
    assert payload["rows"] == [[1], [2]]


def test_query_command_refuses_destructive(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_provider: Any
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr(
        "nl_db.config.default_config_path", lambda: tmp_path / "missing.toml"
    )
    monkeypatch.setattr("nl_db.cli.load_settings", _load_with_logdir(tmp_path))
    patch_provider("```sql\nDELETE FROM t\n```")
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["query", "drop everything", "--db", str(db), "--no-confirm"]
    )
    assert result.exit_code != 0
    # The validator raises; Typer should surface it
    assert "destructive" in result.output.lower() or result.exception is not None


def test_schema_command(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_provider: Any
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr(
        "nl_db.config.default_config_path", lambda: tmp_path / "missing.toml"
    )
    monkeypatch.setattr("nl_db.cli.load_settings", _load_with_logdir(tmp_path))
    patch_provider()  # no LLM calls expected for schema command
    runner = CliRunner()
    result = runner.invoke(cli.app, ["schema", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Table t" in result.output


def test_query_command_cannot_answer(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_provider: Any
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr(
        "nl_db.config.default_config_path", lambda: tmp_path / "missing.toml"
    )
    monkeypatch.setattr("nl_db.cli.load_settings", _load_with_logdir(tmp_path))
    patch_provider(
        "CANNOT_ANSWER: This database has no information about employees."
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.app, ["query", "how many employees?", "--db", str(db), "--no-confirm"]
    )
    assert result.exit_code == 0, result.output
    assert "I can't answer that" in result.output
    assert "employees" in result.output.lower()
    assert "Available tables" in result.output


def test_query_command_clarify_non_interactive(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_provider: Any
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr(
        "nl_db.config.default_config_path", lambda: tmp_path / "missing.toml"
    )
    monkeypatch.setattr("nl_db.cli.load_settings", _load_with_logdir(tmp_path))
    patch_provider("CLARIFY: Do you mean by id or by label?")
    runner = CliRunner()
    # --no-confirm skips the interactive clarification path → exit nonzero
    result = runner.invoke(
        cli.app, ["query", "show me", "--db", str(db), "--no-confirm"]
    )
    assert result.exit_code == 2
    assert "Need more information" in result.output
    assert "by id or by label" in result.output


def test_config_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setattr(
        "nl_db.config.default_config_path", lambda: tmp_path / "missing.toml"
    )
    monkeypatch.setattr("nl_db.cli.load_settings", _load_with_logdir(tmp_path))
    runner = CliRunner()
    result = runner.invoke(cli.app, ["config"])
    assert result.exit_code == 0, result.output
    assert "provider" in result.output
    assert "anthropic" in result.output


# Helpers -------------------------------------------------------------------

def _load_with_logdir(tmp_path: Path) -> Any:
    """load_settings replacement that points log_dir into tmp_path."""
    from nl_db.config import load_settings as real_load

    def _load() -> Any:
        s = real_load(config_path=tmp_path / "missing.toml")
        s.log_dir = tmp_path / "logs"
        return s

    return _load
