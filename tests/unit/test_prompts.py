from __future__ import annotations

from nl_db.llm.provider import ChatResult, Message
from nl_db.prompts.builder import build_sql_prompt, exceeds_budget
from nl_db.prompts.paraphrase import paraphrase_sql
from nl_db.schema.base import Column, Schema, Table


def _schema() -> Schema:
    return Schema(
        dialect="sqlite",
        tables=(
            Table(
                name="users",
                columns=(
                    Column(name="id", type="INTEGER", nullable=False, primary_key=True),
                    Column(name="email", type="TEXT", nullable=False),
                ),
            ),
        ),
    )


def test_build_sql_prompt_includes_system_schema_examples_question() -> None:
    prompt = build_sql_prompt(_schema(), "list all users")
    assert len(prompt.messages) == 2
    sys_msg, user_msg = prompt.messages
    assert sys_msg.role == "system"
    assert "SQLite" in sys_msg.content
    assert user_msg.role == "user"
    assert "Schema:" in user_msg.content
    assert "Table users:" in user_msg.content
    assert "Example 1 question:" in user_msg.content
    assert "Question: list all users" in user_msg.content
    assert prompt.approx_tokens > 0


def test_build_sql_prompt_with_empty_examples() -> None:
    prompt = build_sql_prompt(_schema(), "list all users", examples=())
    assert "Example 1" not in prompt.messages[1].content


def test_exceeds_budget() -> None:
    prompt = build_sql_prompt(_schema(), "list users")
    assert exceeds_budget(prompt, max_tokens=10) is True
    assert exceeds_budget(prompt, max_tokens=10_000) is False


class _FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(text=self._text)


def test_paraphrase_sql_passes_sql_only_no_schema() -> None:
    provider = _FakeProvider(text="Returns each user's email.")
    sentence = paraphrase_sql(provider, "SELECT email FROM users;")
    assert sentence == "Returns each user's email."

    # confirm schema is NOT in the paraphrase prompt
    msgs = provider.calls[0]
    user_content = next(m.content for m in msgs if m.role == "user")
    assert "Table" not in user_content
    assert "SELECT email FROM users;" in user_content
