"""Tests for the lazy-schema agent orchestrator.

Uses a scripted fake provider that returns pre-programmed responses (mixing
text and tool_calls) so we can drive the loop deterministically without
hitting any real LLM.
"""
from __future__ import annotations

from typing import Any

import pytest

from nl_db.agent import (
    LAZY_TOOLS,
    AgentRun,
    LazyAgentError,
    run_lazy_schema,
)
from nl_db.generator import Answer, CannotAnswer, Clarify
from nl_db.llm.provider import (
    ChatResult,
    Message,
    ToolCall,
    ToolDef,
    ToolsNotSupportedError,
)
from nl_db.schema.base import Column, Schema, Table


def _toy_schema() -> Schema:
    return Schema(
        dialect="sqlite",
        tables=(
            Table(
                name="users",
                columns=(
                    Column(name="id", type="INTEGER", nullable=False, primary_key=True),
                    Column(name="name", type="TEXT", nullable=False),
                ),
            ),
            Table(
                name="orders",
                columns=(
                    Column(name="id", type="INTEGER", nullable=False, primary_key=True),
                    Column(name="user_id", type="INTEGER", nullable=False),
                    Column(name="amount", type="REAL", nullable=False),
                ),
            ),
        ),
    )


class ScriptedProvider:
    """Returns pre-programmed ChatResults in order. Records every call's
    tools= and messages= for assertions."""

    name = "scripted"
    model = "scripted-1"
    supports_tools: bool | None = True

    def __init__(self, *results: ChatResult) -> None:
        self._queue = list(results)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        tools: tuple[ToolDef, ...] | None = None,
    ) -> ChatResult:
        self.calls.append(
            {"messages": list(messages), "tools": tools, "temperature": temperature}
        )
        if not self._queue:
            raise AssertionError("ScriptedProvider exhausted")
        return self._queue.pop(0)


def _tool_call(call_id: str, name: str, **args: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


# Happy paths --------------------------------------------------------------

def test_one_shot_answer_without_tool_calls() -> None:
    """The model can answer directly — single iteration, no tool calls."""
    provider = ScriptedProvider(
        ChatResult(text="```sql\nSELECT name FROM users\n```", stop_reason="end_turn"),
    )
    run = run_lazy_schema(provider=provider, schema=_toy_schema(), question="list users")

    assert isinstance(run.outcome, Answer)
    assert "SELECT name FROM users" in run.outcome.sql
    assert run.iterations == 1
    assert run.invocations == []


def test_loop_with_list_tables_then_answer() -> None:
    provider = ScriptedProvider(
        # Iteration 1: model asks for the table list
        ChatResult(
            text="",
            tool_calls=(_tool_call("c1", "list_tables"),),
            stop_reason="tool_use",
        ),
        # Iteration 2: model answers
        ChatResult(
            text="```sql\nSELECT id FROM users\n```",
            stop_reason="end_turn",
        ),
    )
    run = run_lazy_schema(provider=provider, schema=_toy_schema(), question="ids")

    assert isinstance(run.outcome, Answer)
    assert run.iterations == 2
    assert len(run.invocations) == 1
    assert run.invocations[0].name == "list_tables"
    assert run.invocations[0].result == {"tables": ["users", "orders"]}


def test_loop_describe_table_then_answer() -> None:
    provider = ScriptedProvider(
        ChatResult(
            text="",
            tool_calls=(_tool_call("c1", "describe_table", table_name="orders"),),
            stop_reason="tool_use",
        ),
        ChatResult(
            text="```sql\nSELECT SUM(amount) FROM orders\n```",
            stop_reason="end_turn",
        ),
    )
    run = run_lazy_schema(provider=provider, schema=_toy_schema(), question="total spend")

    assert isinstance(run.outcome, Answer)
    assert run.invocations[0].name == "describe_table"
    assert run.invocations[0].result["name"] == "orders"
    cols = [c["name"] for c in run.invocations[0].result["columns"]]
    assert cols == ["id", "user_id", "amount"]


def test_multiple_tool_calls_in_one_iteration() -> None:
    """Anthropic can emit multiple tool_use blocks in one response."""
    provider = ScriptedProvider(
        ChatResult(
            text="Let me look at both tables.",
            tool_calls=(
                _tool_call("c1", "describe_table", table_name="users"),
                _tool_call("c2", "describe_table", table_name="orders"),
            ),
            stop_reason="tool_use",
        ),
        ChatResult(
            text="```sql\nSELECT u.name, SUM(o.amount) FROM users u JOIN orders o ON o.user_id = u.id GROUP BY u.id\n```",
            stop_reason="end_turn",
        ),
    )
    run = run_lazy_schema(provider=provider, schema=_toy_schema(), question="spend by user")

    assert isinstance(run.outcome, Answer)
    assert len(run.invocations) == 2
    assert run.invocations[0].arguments == {"table_name": "users"}
    assert run.invocations[1].arguments == {"table_name": "orders"}


# Three-state outcomes -----------------------------------------------------

def test_cannot_answer_after_lookup() -> None:
    provider = ScriptedProvider(
        ChatResult(
            text="",
            tool_calls=(_tool_call("c1", "list_tables"),),
            stop_reason="tool_use",
        ),
        ChatResult(
            text="CANNOT_ANSWER: This database has no information about employees.",
            stop_reason="end_turn",
        ),
    )
    run = run_lazy_schema(provider=provider, schema=_toy_schema(), question="how many employees")

    assert isinstance(run.outcome, CannotAnswer)
    assert "employees" in run.outcome.reason.lower()


def test_clarify_response() -> None:
    provider = ScriptedProvider(
        ChatResult(
            text="CLARIFY: Do you mean orders by user or by date?",
            stop_reason="end_turn",
        ),
    )
    run = run_lazy_schema(provider=provider, schema=_toy_schema(), question="show orders")

    assert isinstance(run.outcome, Clarify)


# Tool execution edge cases ------------------------------------------------

def test_describe_table_with_unknown_table_returns_error_dict() -> None:
    """When the model asks for a non-existent table, return an error dict
    (so it can recover) rather than raising."""
    provider = ScriptedProvider(
        ChatResult(
            text="",
            tool_calls=(_tool_call("c1", "describe_table", table_name="employees"),),
            stop_reason="tool_use",
        ),
        ChatResult(
            text="CANNOT_ANSWER: No employees table.",
            stop_reason="end_turn",
        ),
    )
    run = run_lazy_schema(provider=provider, schema=_toy_schema(), question="employees?")

    assert isinstance(run.outcome, CannotAnswer)
    assert "error" in run.invocations[0].result
    assert run.invocations[0].result["available_tables"] == ["users", "orders"]


def test_unknown_tool_returns_error_dict_does_not_raise() -> None:
    provider = ScriptedProvider(
        ChatResult(
            text="",
            tool_calls=(_tool_call("c1", "drop_database"),),
            stop_reason="tool_use",
        ),
        ChatResult(
            text="CANNOT_ANSWER: That's not a real tool.",
            stop_reason="end_turn",
        ),
    )
    run = run_lazy_schema(provider=provider, schema=_toy_schema(), question="?")

    # The model sees an error dict and chooses CANNOT_ANSWER; no crash.
    assert isinstance(run.outcome, CannotAnswer)
    assert "Unknown tool" in run.invocations[0].result["error"]


# Provider capability + failure modes --------------------------------------

def test_provider_explicitly_unsupporting_tools_raises() -> None:
    class NoToolsProvider:
        name = "no-tools"
        model = "x"
        supports_tools = False

        def chat(self, *args: Any, **kwargs: Any) -> ChatResult:
            raise AssertionError("chat() should not have been called")

    with pytest.raises(ToolsNotSupportedError):
        run_lazy_schema(
            provider=NoToolsProvider(),  # type: ignore[arg-type]
            schema=_toy_schema(),
            question="?",
        )


def test_loop_aborts_after_max_iterations() -> None:
    """If the model keeps calling tools forever, we give up cleanly."""
    looping_call = ChatResult(
        text="",
        tool_calls=(_tool_call("c1", "list_tables"),),
        stop_reason="tool_use",
    )
    provider = ScriptedProvider(*([looping_call] * 10))
    with pytest.raises(LazyAgentError, match="did not finish"):
        run_lazy_schema(
            provider=provider,
            schema=_toy_schema(),
            question="loop forever",
            max_iterations=3,
        )


def test_empty_text_with_no_tool_calls_raises() -> None:
    """If the model gives up — no tools, no text — surface it as an error
    so the caller can fall back rather than getting a confusing empty
    Answer."""
    provider = ScriptedProvider(
        ChatResult(text="", stop_reason="end_turn"),
    )
    with pytest.raises(LazyAgentError, match="empty text"):
        run_lazy_schema(provider=provider, schema=_toy_schema(), question="?")


# Wire shape ---------------------------------------------------------------

def test_provider_receives_lazy_tools_on_every_iteration() -> None:
    provider = ScriptedProvider(
        ChatResult(
            text="",
            tool_calls=(_tool_call("c1", "list_tables"),),
            stop_reason="tool_use",
        ),
        ChatResult(text="```sql\nSELECT 1\n```", stop_reason="end_turn"),
    )
    run_lazy_schema(provider=provider, schema=_toy_schema(), question="?")

    for call in provider.calls:
        tool_names = [t.name for t in (call["tools"] or ())]
        assert "list_tables" in tool_names
        assert "describe_table" in tool_names


def test_schema_is_not_injected_into_prompt() -> None:
    """Key contract: in lazy mode the schema is NEVER part of the prompt;
    the model must use tools to access it."""
    provider = ScriptedProvider(
        ChatResult(text="```sql\nSELECT 1\n```", stop_reason="end_turn"),
    )
    run_lazy_schema(provider=provider, schema=_toy_schema(), question="?")

    user_msg = next(
        m.content for m in provider.calls[0]["messages"] if m.role == "user"
    )
    assert "Table users" not in user_msg
    assert "INTEGER" not in user_msg
    assert "Question:" in user_msg


def test_agent_run_has_lazy_tools_constant_in_module() -> None:
    """LAZY_TOOLS is the canonical list — make sure it stays in sync with
    what the orchestrator actually exposes (would catch a future tool added
    to one path but not the other)."""
    names = {t.name for t in LAZY_TOOLS}
    assert names == {"list_tables", "describe_table"}


def test_agent_run_dataclass_default_invocations_is_independent_list() -> None:
    """Catch the classic mutable-default-argument bug."""
    a = AgentRun(outcome=Clarify(question="?"))
    b = AgentRun(outcome=Clarify(question="?"))
    a.invocations.append("x")  # type: ignore[arg-type]
    assert b.invocations == []
