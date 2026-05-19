from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .executor import QueryExecutor, QueryResult, SQLiteExecutor
from .generator import generate_sql
from .llm.provider import LLMProvider
from .prompts.builder import BuiltPrompt, build_sql_prompt, exceeds_budget
from .prompts.paraphrase import paraphrase_sql
from .schema.base import Schema
from .schema.cache import SchemaCache
from .schema.sqlite import SQLiteSchemaExtractor
from .validator import ValidationResult, validate_sql


@dataclass
class PipelineOutput:
    question: str
    sql_raw: str
    sql_final: str  # post-validation (may include auto-LIMIT)
    paraphrase: str | None
    result: QueryResult | None
    prompt: BuiltPrompt
    validation: ValidationResult
    auto_limit_applied: bool
    is_destructive: bool
    confirmed: bool
    skipped_reason: str | None = None


# Confirmation callback: receives the final SQL + paraphrase, returns True to proceed.
ConfirmFn = Callable[[str, str | None], bool]


def _auto_confirm(_sql: str, _paraphrase: str | None) -> bool:
    return True


class Pipeline:
    """NL question → SQL → validate → paraphrase → confirm → execute → result.

    SQLite-specific in v1, but constructed from a generic LLMProvider so the
    LLM side is provider-agnostic. The confirmation callback abstracts UX so
    the same pipeline serves CLI (interactive prompt), MCP (return SQL to
    host model), and non-interactive (`--no-confirm` / auto_confirm).
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        db_path: Path,
        max_rows: int = 1000,
        timeout_s: float = 10.0,
        max_prompt_tokens: int = 8000,
        schema_cache: SchemaCache | None = None,
        executor: QueryExecutor | None = None,
        paraphrase: bool = True,
    ) -> None:
        self._provider = provider
        self._db_path = db_path
        self._max_rows = max_rows
        self._max_prompt_tokens = max_prompt_tokens
        self._cache = schema_cache or SchemaCache()
        self._executor = executor or SQLiteExecutor(
            db_path, timeout_s=timeout_s, max_rows=max_rows
        )
        self._paraphrase = paraphrase

    def schema(self) -> Schema:
        return self._cache.get(
            self._db_path, lambda: SQLiteSchemaExtractor.from_path(self._db_path).extract()
        )

    def run(
        self,
        question: str,
        *,
        allow_writes: bool = False,
        confirm: ConfirmFn | None = None,
    ) -> PipelineOutput:
        schema = self.schema()
        prompt = build_sql_prompt(schema, question)
        if exceeds_budget(prompt, self._max_prompt_tokens):
            # Soft warning only — the LLM may still handle it. The caller can
            # inspect prompt.approx_tokens against max_prompt_tokens if it
            # wants to surface this to the user.
            pass

        sql_raw = generate_sql(self._provider, prompt)
        validation = validate_sql(
            sql_raw,
            dialect=schema.dialect,
            allow_writes=allow_writes,
            max_rows=self._max_rows,
        )

        paraphrase: str | None = None
        if self._paraphrase:
            paraphrase = paraphrase_sql(self._provider, validation.sql)

        confirm_fn = confirm or _auto_confirm
        confirmed = confirm_fn(validation.sql, paraphrase)

        result: QueryResult | None = None
        skipped_reason: str | None = None
        if confirmed:
            result = self._executor.execute(validation.sql)
        else:
            skipped_reason = "user declined to run the SQL"

        return PipelineOutput(
            question=question,
            sql_raw=sql_raw,
            sql_final=validation.sql,
            paraphrase=paraphrase,
            result=result,
            prompt=prompt,
            validation=validation,
            auto_limit_applied=validation.auto_limit_applied,
            is_destructive=validation.is_destructive,
            confirmed=confirmed,
            skipped_reason=skipped_reason,
        )


