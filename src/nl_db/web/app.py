"""Streamlit playground for nl-db.

Two purposes:
1. Edit the configuration in a live UI (provider, model, key, DB, limits, all
   the tuning knobs the Pipeline now exposes).
2. Run NL → SQL → result interactively without touching the CLI.

The UI never writes to ~/.config/nl-db/config.toml or .env — config edits are
session-scoped (st.session_state). Restart and your edits are gone. This keeps
the UI a safe sandbox for experimentation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from nl_db.config import Settings, load_settings
from nl_db.llm.anthropic_provider import AnthropicProvider
from nl_db.llm.openai_compatible import OpenAICompatibleProvider
from nl_db.llm.openai_provider import OpenAIProvider
from nl_db.llm.provider import LLMProvider
from nl_db.pipeline import Pipeline, PipelineOutput
from nl_db.schema.base import render_for_prompt
from nl_db.schema.sqlite import SQLiteSchemaExtractor
from nl_db.validator import SQLValidationError, validate_sql

st.set_page_config(
    page_title="nl-db playground",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state plumbing
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    ts: float
    question: str
    sql: str
    paraphrase: str | None
    row_count: int | None
    success: bool
    error: str | None = None


def _init_state() -> None:
    if "initialized" in st.session_state:
        return
    base = load_settings()
    st.session_state.initialized = True
    st.session_state.provider_name = base.provider.name
    st.session_state.model = base.provider.model
    st.session_state.base_url = base.provider.base_url or ""
    st.session_state.api_key = (
        base.provider.api_key.get_secret_value() if base.provider.api_key else ""
    )
    st.session_state.db_path = ""
    st.session_state.max_rows = base.limits.max_rows
    st.session_state.timeout_s = base.limits.timeout_s
    st.session_state.max_prompt_tokens = base.limits.max_prompt_tokens
    st.session_state.temperature = 0.0
    st.session_state.max_output_tokens = 512
    st.session_state.paraphrase_enabled = True
    st.session_state.paraphrase_temperature = 0.0
    st.session_state.paraphrase_max_output_tokens = 128
    st.session_state.auto_limit = True
    st.session_state.num_few_shot = -1  # -1 sentinel = use default (all)
    st.session_state.allow_writes = False
    st.session_state.history: list[HistoryEntry] = []
    st.session_state.last_output: PipelineOutput | None = None
    st.session_state.edited_sql: str | None = None


_init_state()


# ---------------------------------------------------------------------------
# Provider/pipeline construction from session state
# ---------------------------------------------------------------------------

def _build_provider() -> LLMProvider:
    name = st.session_state.provider_name
    model = st.session_state.model
    key = st.session_state.api_key
    if name == "anthropic":
        if not key:
            raise RuntimeError("Anthropic API key is required.")
        return AnthropicProvider(model=model, api_key=key)
    if name == "openai":
        if not key:
            raise RuntimeError("OpenAI API key is required.")
        return OpenAIProvider(model=model, api_key=key)
    if name == "openai_compatible":
        base_url = st.session_state.base_url
        if not base_url:
            raise RuntimeError("Base URL required for openai_compatible.")
        return OpenAICompatibleProvider(
            model=model, base_url=base_url, api_key=key or None
        )
    raise RuntimeError(f"Unknown provider: {name}")


def _build_pipeline() -> Pipeline:
    db_path = Path(st.session_state.db_path).expanduser()
    if not db_path.exists():
        raise RuntimeError(f"Database not found: {db_path}")
    n_few_shot: int | None = (
        None if st.session_state.num_few_shot == -1 else st.session_state.num_few_shot
    )
    return Pipeline(
        provider=_build_provider(),
        db_path=db_path,
        max_rows=int(st.session_state.max_rows),
        timeout_s=float(st.session_state.timeout_s),
        max_prompt_tokens=int(st.session_state.max_prompt_tokens),
        paraphrase=bool(st.session_state.paraphrase_enabled),
        temperature=float(st.session_state.temperature),
        max_output_tokens=int(st.session_state.max_output_tokens),
        paraphrase_temperature=float(st.session_state.paraphrase_temperature),
        paraphrase_max_output_tokens=int(st.session_state.paraphrase_max_output_tokens),
        auto_limit=bool(st.session_state.auto_limit),
        num_few_shot=n_few_shot,
    )


# ---------------------------------------------------------------------------
# Sidebar — all the knobs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("nl-db")
    st.caption("session-scoped config — nothing is written to disk")

    with st.expander("Database", expanded=True):
        st.text_input(
            "SQLite path",
            key="db_path",
            placeholder="/abs/path/to/your.db",
            help="Absolute path to a SQLite database file.",
        )

    with st.expander("Provider", expanded=True):
        st.selectbox(
            "Provider",
            options=["anthropic", "openai", "openai_compatible"],
            key="provider_name",
        )
        st.text_input("Model", key="model")
        if st.session_state.provider_name == "openai_compatible":
            st.text_input(
                "Base URL",
                key="base_url",
                placeholder="http://localhost:8080/v1",
                help="Endpoint speaking OpenAI /v1/chat/completions.",
            )
        st.text_input(
            "API key",
            key="api_key",
            type="password",
            help=(
                "Stored only in the running session — not written to disk. "
                "Most local OpenAI-compatible servers accept any string."
            ),
        )

    with st.expander("Generation", expanded=False):
        st.slider("Temperature (SQL gen)", 0.0, 1.5, key="temperature", step=0.05)
        st.number_input(
            "Max output tokens (SQL gen)",
            min_value=64,
            max_value=4096,
            step=64,
            key="max_output_tokens",
        )
        st.checkbox("Run paraphrase pass", key="paraphrase_enabled")
        if st.session_state.paraphrase_enabled:
            st.slider(
                "Temperature (paraphrase)",
                0.0,
                1.5,
                key="paraphrase_temperature",
                step=0.05,
            )
            st.number_input(
                "Max output tokens (paraphrase)",
                min_value=32,
                max_value=512,
                step=16,
                key="paraphrase_max_output_tokens",
            )
        st.slider(
            "Few-shot examples (-1 = default, 0 = none)",
            min_value=-1,
            max_value=10,
            key="num_few_shot",
        )

    with st.expander("Execution", expanded=False):
        st.number_input(
            "Max rows returned",
            min_value=1,
            max_value=100_000,
            step=50,
            key="max_rows",
        )
        st.checkbox("Auto-inject LIMIT", key="auto_limit")
        st.slider(
            "Statement timeout (s)",
            min_value=1.0,
            max_value=60.0,
            step=0.5,
            key="timeout_s",
        )
        st.number_input(
            "Max prompt tokens (soft warning)",
            min_value=512,
            max_value=200_000,
            step=512,
            key="max_prompt_tokens",
        )

    with st.expander("Safety", expanded=False):
        st.checkbox(
            "Allow writes (INSERT/UPDATE/DELETE/DROP/ALTER)",
            key="allow_writes",
            help="Off by default. The validator refuses destructive SQL unless this is checked.",
        )
        if st.session_state.allow_writes:
            st.warning(
                "Writes are enabled. Destructive SQL will execute against the database.",
                icon="⚠️",
            )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Natural language → SQL playground")
st.caption(
    "Schema-first prompting · SQL transparency · read-only by default · "
    "every LLM call goes through the provider Protocol — no vendor lock-in"
)

tab_query, tab_schema, tab_history, tab_about = st.tabs(
    ["Query", "Schema", "History", "About"]
)


# --- Query tab -------------------------------------------------------------

with tab_query:
    if not st.session_state.db_path:
        st.info("Set a SQLite path in the sidebar to get started.", icon="📁")
    else:
        col_q, col_btn = st.columns([5, 1], vertical_alignment="bottom")
        with col_q:
            question = st.text_area(
                "Question",
                key="question_input",
                placeholder='"How much did each user spend last month?"',
                height=80,
            )
        with col_btn:
            generate_clicked = st.button(
                "Generate SQL", type="primary", use_container_width=True
            )

        if generate_clicked and question.strip():
            with st.spinner("Calling LLM..."):
                try:
                    pipeline = _build_pipeline()
                    schema = pipeline.schema()
                    from nl_db.generator import generate_sql
                    from nl_db.prompts.builder import build_sql_prompt
                    from nl_db.prompts.paraphrase import paraphrase_sql

                    n_few_shot: int | None = (
                        None
                        if st.session_state.num_few_shot == -1
                        else st.session_state.num_few_shot
                    )
                    examples = pipeline._select_examples(schema)  # noqa: SLF001
                    prompt = build_sql_prompt(schema, question, examples=examples)
                    raw_sql = generate_sql(
                        pipeline._provider,  # noqa: SLF001
                        prompt,
                        temperature=float(st.session_state.temperature),
                        max_output_tokens=int(st.session_state.max_output_tokens),
                    )
                    validation = validate_sql(
                        raw_sql,
                        dialect=schema.dialect,
                        allow_writes=bool(st.session_state.allow_writes),
                        max_rows=(
                            int(st.session_state.max_rows)
                            if st.session_state.auto_limit
                            else None
                        ),
                    )
                    paraphrase = None
                    if st.session_state.paraphrase_enabled:
                        paraphrase = paraphrase_sql(
                            pipeline._provider,  # noqa: SLF001
                            validation.sql,
                            temperature=float(
                                st.session_state.paraphrase_temperature
                            ),
                            max_output_tokens=int(
                                st.session_state.paraphrase_max_output_tokens
                            ),
                        )
                    st.session_state.last_output = {
                        "question": question,
                        "sql_raw": raw_sql,
                        "sql_final": validation.sql,
                        "paraphrase": paraphrase,
                        "is_destructive": validation.is_destructive,
                        "auto_limit_applied": validation.auto_limit_applied,
                        "approx_prompt_tokens": prompt.approx_tokens,
                    }
                    st.session_state.edited_sql = validation.sql
                except SQLValidationError as e:
                    st.error(f"Validation refused this SQL: {e}", icon="🛑")
                    st.session_state.last_output = None
                except Exception as e:  # noqa: BLE001
                    st.error(f"{type(e).__name__}: {e}", icon="❌")
                    st.session_state.last_output = None

        out = st.session_state.last_output
        if out:
            c1, c2, c3 = st.columns(3)
            c1.metric("Prompt tokens (approx)", out["approx_prompt_tokens"])
            c2.metric(
                "Auto-LIMIT",
                "applied" if out["auto_limit_applied"] else "skipped",
            )
            c3.metric(
                "Type", "destructive" if out["is_destructive"] else "read-only"
            )

            st.subheader("Generated SQL")
            st.caption("Edit if you'd like before running.")
            st.session_state.edited_sql = st.text_area(
                "SQL",
                value=st.session_state.edited_sql or out["sql_final"],
                height=200,
                label_visibility="collapsed",
                key="sql_edit_box",
            )

            if out["paraphrase"]:
                st.subheader("In plain English")
                st.success(out["paraphrase"])

            run_col, _ = st.columns([1, 5])
            with run_col:
                run_clicked = st.button(
                    "Run SQL",
                    type="primary",
                    use_container_width=True,
                    disabled=not st.session_state.edited_sql,
                )

            if run_clicked:
                try:
                    pipeline = _build_pipeline()
                    sql_to_run = st.session_state.edited_sql or out["sql_final"]
                    # Re-validate edited SQL with the same allow_writes setting.
                    revalidation = validate_sql(
                        sql_to_run,
                        dialect=pipeline.schema().dialect,
                        allow_writes=bool(st.session_state.allow_writes),
                        max_rows=(
                            int(st.session_state.max_rows)
                            if st.session_state.auto_limit
                            else None
                        ),
                    )
                    result = pipeline._executor.execute(revalidation.sql)  # noqa: SLF001
                    st.subheader("Result")
                    if result.columns:
                        df = pd.DataFrame(result.rows, columns=list(result.columns))
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Query produced no columns.")
                    meta_a, meta_b, meta_c = st.columns(3)
                    meta_a.metric("Rows", result.row_count)
                    meta_b.metric(
                        "Truncated", "yes" if result.truncated else "no"
                    )
                    meta_c.metric(
                        "Auto-LIMIT",
                        "applied" if revalidation.auto_limit_applied else "skipped",
                    )

                    with st.expander("Raw JSON"):
                        st.code(
                            json.dumps(
                                {
                                    "columns": list(result.columns),
                                    "rows": [list(r) for r in result.rows],
                                    "row_count": result.row_count,
                                    "truncated": result.truncated,
                                },
                                default=str,
                                indent=2,
                            ),
                            language="json",
                        )

                    st.session_state.history.insert(
                        0,
                        HistoryEntry(
                            ts=time.time(),
                            question=out["question"],
                            sql=revalidation.sql,
                            paraphrase=out["paraphrase"],
                            row_count=result.row_count,
                            success=True,
                        ),
                    )
                except SQLValidationError as e:
                    st.error(f"Validation refused this SQL: {e}", icon="🛑")
                    st.session_state.history.insert(
                        0,
                        HistoryEntry(
                            ts=time.time(),
                            question=out["question"],
                            sql=st.session_state.edited_sql or out["sql_final"],
                            paraphrase=out["paraphrase"],
                            row_count=None,
                            success=False,
                            error=str(e),
                        ),
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"{type(e).__name__}: {e}", icon="❌")
                    st.session_state.history.insert(
                        0,
                        HistoryEntry(
                            ts=time.time(),
                            question=out["question"],
                            sql=st.session_state.edited_sql or out["sql_final"],
                            paraphrase=out["paraphrase"],
                            row_count=None,
                            success=False,
                            error=f"{type(e).__name__}: {e}",
                        ),
                    )


# --- Schema tab ------------------------------------------------------------

with tab_schema:
    if not st.session_state.db_path:
        st.info("Set a SQLite path in the sidebar.", icon="📁")
    else:
        try:
            schema = SQLiteSchemaExtractor.from_path(
                Path(st.session_state.db_path).expanduser()
            ).extract()
        except Exception as e:  # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}", icon="❌")
        else:
            st.caption(
                f"{len(schema.tables)} table(s) in {st.session_state.db_path}"
            )
            for table in schema.tables:
                with st.expander(f"📋 {table.name}", expanded=False):
                    rows: list[dict[str, Any]] = []
                    fk_by_col = {fk.column: fk for fk in table.foreign_keys}
                    for col in table.columns:
                        fk = fk_by_col.get(col.name)
                        rows.append(
                            {
                                "column": col.name,
                                "type": col.type,
                                "nullable": col.nullable,
                                "primary_key": col.primary_key,
                                "default": col.default,
                                "references": (
                                    f"{fk.references_table}.{fk.references_column}"
                                    if fk
                                    else None
                                ),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )
            with st.expander("Prompt rendering (what the LLM sees)"):
                st.code(render_for_prompt(schema), language="text")


# --- History tab -----------------------------------------------------------

with tab_history:
    if not st.session_state.history:
        st.caption("No queries run in this session yet.")
    else:
        for i, entry in enumerate(st.session_state.history):
            badge = "✅" if entry.success else "❌"
            with st.expander(
                f"{badge} {entry.question[:80]} — "
                f"{time.strftime('%H:%M:%S', time.localtime(entry.ts))}",
                expanded=(i == 0),
            ):
                st.code(entry.sql, language="sql")
                if entry.paraphrase:
                    st.caption(entry.paraphrase)
                if entry.success:
                    st.write(f"{entry.row_count} row(s) returned")
                else:
                    st.error(entry.error or "unknown error")


# --- About tab -------------------------------------------------------------

with tab_about:
    st.markdown(
        """
        ### About this playground

        This Streamlit UI is a *sandbox* on top of the nl-db pipeline.

        - **Config is session-scoped.** Nothing in the sidebar is written to disk.
          To persist, edit `~/.config/nl-db/config.toml` or your `.env`.
        - **API keys** typed here live only in this Streamlit process.
        - **Every LLM call** goes through the same `LLMProvider` Protocol the
          CLI and the MCP server use.
        - **Read-only by default.** The validator (sqlglot parse-tree) refuses
          destructive statements unless *Allow writes* is checked in the sidebar.

        Same pipeline also runs as:
        - **CLI:** `uv run nl-db query "..."`
        - **MCP server:** `uv run nl-db-mcp --db ...` (for Claude Desktop, Cursor)
        """
    )

    s: Settings = load_settings()
    st.caption("On-disk defaults (`load_settings()`):")
    st.json(
        {
            "provider": s.provider.name,
            "model": s.provider.model,
            "base_url": s.provider.base_url,
            "max_rows": s.limits.max_rows,
            "timeout_s": s.limits.timeout_s,
            "max_prompt_tokens": s.limits.max_prompt_tokens,
            "log_dir": str(s.log_dir),
        }
    )
