"""Lazy-schema agent orchestrator.

Runs a tool-use loop against the configured LLM, exposing two tools that
let the model look up schema details on demand instead of having the full
schema injected into the prompt.

Tools the model sees:
  - list_tables(): returns ["users", "orders", ...]
  - describe_table(table_name): returns columns + FKs for one table

Outcome shape is identical to the schema-injection path: the final result
is a `GenerationOutcome` (Answer / CannotAnswer / Clarify), parsed from
the model's last text response after it stops calling tools.

When the configured provider doesn't support tools, this module raises
`ToolsNotSupportedError`. The caller (Pipeline) catches it and falls
back to schema injection — never a silent failure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .generator import GenerationOutcome, parse_outcome
from .llm.provider import (
    LLMProvider,
    Message,
    ToolCall,
    ToolDef,
    ToolsNotSupportedError,
)
from .schema.base import Schema

LAZY_SYSTEM = """You translate plain-English questions into SQL for a SQLite database.

You do NOT have the schema in your context. Use the provided tools to look it up:

- `list_tables()` returns every user-created table in the database.
- `describe_table(table_name)` returns the columns (with types and nullability), primary keys, and foreign keys of one specific table.

Workflow:
1. If you don't yet know what tables exist, call `list_tables()` first.
2. Call `describe_table(...)` on each table you'll need for the question.
3. Once you have enough information, respond with exactly ONE of three formats:

   a. ```sql
      <a single SELECT or CTE-feeding-SELECT statement>
      ```
   b. CANNOT_ANSWER: <one short plain-English sentence>
      Use this when the schema simply doesn't contain the data needed.
   c. CLARIFY: <one short plain-English follow-up question>
      Use this when the question is genuinely ambiguous and one clarifying
      question would let you proceed.

Rules for the SQL format:
- Wrap in a fenced ```sql ... ``` block. No prose, no comments, no second statement.
- SELECT-only. Never INSERT/UPDATE/DELETE/DROP/ALTER.
- Use only the tables and columns you actually looked up. Do not invent identifiers.
- SQLite-flavored: `strftime`, `date('now', ...)`, `julianday`, `||` for concatenation.
- ORDER BY a meaningful column when listing rows.
- Prefer explicit JOINs over implicit cross-products.

Call tools only as many times as you need — don't dump the whole schema if the
question only touches one or two tables. If the question is simple ("count
the users") you may be able to answer after just one tool call.
"""


def _list_tables_tool_def() -> ToolDef:
    return ToolDef(
        name="list_tables",
        description=(
            "Return every user-created table name in the connected database. "
            "Call this first if you don't already know what tables exist."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def _describe_table_tool_def() -> ToolDef:
    return ToolDef(
        name="describe_table",
        description=(
            "Return the schema for one specific table: columns with types and "
            "nullability, primary keys, and foreign-key references."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Exact table name from list_tables().",
                }
            },
            "required": ["table_name"],
            "additionalProperties": False,
        },
    )


LAZY_TOOLS: tuple[ToolDef, ...] = (
    _list_tables_tool_def(),
    _describe_table_tool_def(),
)


@dataclass
class ToolInvocation:
    """One tool round-trip — what the model asked for and what we returned.

    Useful for the Debug expander so users can inspect the trace of lazy-schema
    reasoning.
    """

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    call_id: str = ""


@dataclass
class AgentRun:
    """Outcome of one run of the lazy-schema agent.

    `outcome` is the final GenerationOutcome (same shape the schema-injection
    path produces). `invocations` is the tool-call trace, in order. `iterations`
    is the number of LLM round-trips it took (one per chat() call).
    """

    outcome: GenerationOutcome
    invocations: list[ToolInvocation] = field(default_factory=list)
    iterations: int = 0


class LazyAgentError(RuntimeError):
    """Raised when the lazy-schema loop fails in a way the caller should know
    about (e.g. exceeded max iterations, model called an unknown tool)."""


def run_lazy_schema(
    *,
    provider: LLMProvider,
    schema: Schema,
    question: str,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    max_iterations: int = 8,
) -> AgentRun:
    """Run a tool-use loop, returning a GenerationOutcome + trace.

    The Schema is the source of truth for tool answers — we don't query the
    DB during the loop. The schema cache lives at a higher layer.

    Raises:
      ToolsNotSupportedError: provider explicitly rejected tools=. Caller
        should fall back to schema injection.
      LazyAgentError: loop ran past max_iterations, or the model called a
        tool we don't expose.
    """
    if provider.supports_tools is False:
        raise ToolsNotSupportedError(
            f"Provider {provider.name!r} reports tool-calling unsupported."
        )

    messages: list[Message] = [
        Message(role="system", content=LAZY_SYSTEM),
        Message(role="user", content=f"Question: {question}"),
    ]
    invocations: list[ToolInvocation] = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        result = provider.chat(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=LAZY_TOOLS,
        )

        if not result.tool_calls:
            # Model finished — text is the GenerationOutcome wire format.
            # If text is empty AND there were no tool calls, that's a model
            # quirk (or empty content from a reasoning model that exhausted
            # its budget on internal thinking) — bubble up as LazyAgentError
            # so the caller can fall back rather than silently fail.
            if not result.text.strip():
                raise LazyAgentError(
                    f"Model returned empty text and no tool calls after "
                    f"{iterations} iteration(s). Try increasing "
                    f"max_output_tokens or switching to a different model."
                )
            outcome = parse_outcome(result.text)
            return AgentRun(
                outcome=outcome,
                invocations=invocations,
                iterations=iterations,
            )

        # Append the assistant turn that requested the tools, then each tool
        # result. Anthropic's translator groups consecutive role='tool'
        # Messages into one tool_result-containing user message; OpenAI's
        # accepts them individually.
        # The assistant message preserves the model's text-before-tool-use
        # commentary (if any), but for chat-completions wire we only need
        # content as a string; the SDKs round-trip tool_use identifiers
        # internally so we just emit a placeholder.
        messages.append(Message(role="assistant", content=result.text or ""))

        for call in result.tool_calls:
            tool_output = _execute_tool(call, schema)
            invocations.append(
                ToolInvocation(
                    name=call.name,
                    arguments=call.arguments,
                    result=tool_output,
                    call_id=call.id,
                )
            )
            messages.append(
                Message(
                    role="tool",
                    content=json.dumps(tool_output),
                    tool_call_id=call.id,
                    tool_name=call.name,
                )
            )

    raise LazyAgentError(
        f"Agent did not finish after {max_iterations} iterations. "
        f"Last seen tool calls: {[i.name for i in invocations[-3:]]}."
    )


def _execute_tool(call: ToolCall, schema: Schema) -> dict[str, Any]:
    """Run one tool against the schema. Returns a JSON-serializable dict.

    Unknown tools return an error dict (we don't raise — that would derail
    the model when a simple "not a real tool" response would let it recover).
    """
    if call.name == "list_tables":
        return {"tables": list(schema.table_names())}
    if call.name == "describe_table":
        table_name = call.arguments.get("table_name") if call.arguments else None
        if not isinstance(table_name, str):
            return {"error": "describe_table requires a string `table_name` argument."}
        table = schema.table(table_name)
        if table is None:
            return {
                "error": f"No table named {table_name!r}.",
                "available_tables": list(schema.table_names()),
            }
        return {
            "name": table.name,
            "columns": [
                {
                    "name": c.name,
                    "type": c.type,
                    "nullable": c.nullable,
                    "primary_key": c.primary_key,
                    "default": c.default,
                }
                for c in table.columns
            ],
            "foreign_keys": [
                {
                    "column": fk.column,
                    "references_table": fk.references_table,
                    "references_column": fk.references_column,
                }
                for fk in table.foreign_keys
            ],
        }
    return {"error": f"Unknown tool {call.name!r}. Available: list_tables, describe_table."}
