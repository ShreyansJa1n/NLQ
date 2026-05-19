from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .executor import QueryExecutor, QueryResult, SQLiteExecutor
from .generator import (
    Answer,
    CannotAnswer,
    Clarify,
    GenerationOutcome,
    generate_outcome,
)
from .llm.provider import LLMProvider
from .prompts.builder import BuiltPrompt, build_sql_prompt, exceeds_budget
from .prompts.examples import FewShotExample, few_shot_for
from .prompts.paraphrase import paraphrase_sql
from .schema.base import Schema
from .schema.cache import SchemaCache
from .schema.sqlite import SQLiteSchemaExtractor
from .validator import ValidationResult, validate_sql


@dataclass
class PipelineOutput:
    """Result of one Pipeline.run() call.

    `outcome` is always populated and tells you what happened. Fields under
    "When outcome is Answer" are populated only for the Answer branch. The
    `state` property gives a stable string for callers that don't want to
    isinstance-check.
    """

    question: str
    outcome: GenerationOutcome
    prompt: BuiltPrompt

    # When outcome is Answer:
    sql_raw: str | None = None
    sql_final: str | None = None
    paraphrase: str | None = None
    result: QueryResult | None = None
    validation: ValidationResult | None = None
    auto_limit_applied: bool = False
    is_destructive: bool = False
    confirmed: bool = False
    skipped_reason: str | None = None

    @property
    def state(self) -> str:
        if isinstance(self.outcome, Answer):
            return "ANSWER"
        if isinstance(self.outcome, CannotAnswer):
            return "CANNOT_ANSWER"
        if isinstance(self.outcome, Clarify):
            return "CLARIFY"
        return "UNKNOWN"


# Confirmation callback: receives the final SQL + paraphrase, returns True to proceed.
ConfirmFn = Callable[[str, str | None], bool]


def _auto_confirm(_sql: str, _paraphrase: str | None) -> bool:
    return True


class Pipeline:
    """NL question → outcome → (if Answer) SQL → validate → paraphrase → confirm → execute.

    SQLite-specific in v1, but constructed from a generic LLMProvider so the
    LLM side is provider-agnostic. The confirmation callback abstracts UX so
    the same pipeline serves CLI (interactive prompt), MCP (return SQL to
    host model), and non-interactive (`--no-confirm` / auto_confirm).

    The pipeline returns a three-state outcome: Answer (we have SQL and can
    run it), CannotAnswer (the schema doesn't cover the question), or
    Clarify (the question is ambiguous and we need a follow-up). Only the
    Answer branch hits the validator / paraphrase / executor.
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
        # Generation tuning knobs (exposed for the Streamlit playground)
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        paraphrase_temperature: float = 0.0,
        paraphrase_max_output_tokens: int = 128,
        auto_limit: bool = True,
        num_few_shot: int | None = None,
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
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._paraphrase_temperature = paraphrase_temperature
        self._paraphrase_max_output_tokens = paraphrase_max_output_tokens
        self._auto_limit = auto_limit
        self._num_few_shot = num_few_shot

    def schema(self) -> Schema:
        return self._cache.get(
            self._db_path, lambda: SQLiteSchemaExtractor.from_path(self._db_path).extract()
        )

    def _select_examples(self, schema: Schema) -> tuple[FewShotExample, ...] | None:
        if self._num_few_shot is None:
            return None  # builder will use default
        if self._num_few_shot <= 0:
            return ()
        return few_shot_for(schema.dialect)[: self._num_few_shot]

    def run(
        self,
        question: str,
        *,
        allow_writes: bool = False,
        confirm: ConfirmFn | None = None,
    ) -> PipelineOutput:
        schema = self.schema()
        prompt = build_sql_prompt(
            schema, question, examples=self._select_examples(schema)
        )
        if exceeds_budget(prompt, self._max_prompt_tokens):
            # Soft warning only — the LLM may still handle it. The caller can
            # inspect prompt.approx_tokens against max_prompt_tokens if it
            # wants to surface this to the user.
            pass

        outcome = generate_outcome(
            self._provider,
            prompt,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )

        # CannotAnswer + Clarify short-circuit. For CannotAnswer we inject
        # available_tables from the live schema so callers don't have to
        # re-fetch it to suggest alternatives.
        if isinstance(outcome, CannotAnswer):
            outcome = CannotAnswer(
                reason=outcome.reason,
                available_tables=schema.table_names(),
            )
            return PipelineOutput(question=question, outcome=outcome, prompt=prompt)

        if isinstance(outcome, Clarify):
            return PipelineOutput(question=question, outcome=outcome, prompt=prompt)

        # Answer branch — validate, paraphrase, confirm, execute.
        validation = validate_sql(
            outcome.sql,
            dialect=schema.dialect,
            allow_writes=allow_writes,
            max_rows=self._max_rows if self._auto_limit else None,
        )

        paraphrase: str | None = None
        if self._paraphrase:
            paraphrase = paraphrase_sql(
                self._provider,
                validation.sql,
                temperature=self._paraphrase_temperature,
                max_output_tokens=self._paraphrase_max_output_tokens,
            )

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
            outcome=outcome,
            prompt=prompt,
            sql_raw=outcome.sql,
            sql_final=validation.sql,
            paraphrase=paraphrase,
            result=result,
            validation=validation,
            auto_limit_applied=validation.auto_limit_applied,
            is_destructive=validation.is_destructive,
            confirmed=confirmed,
            skipped_reason=skipped_reason,
        )
