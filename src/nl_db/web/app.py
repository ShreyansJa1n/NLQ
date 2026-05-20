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

import anthropic
import httpx
import openai
import pandas as pd
import streamlit as st

from nl_db.config import (
    GenerationConfig,
    LimitsConfig,
    ProviderConfig,
    Settings,
    default_config_path,
    load_settings,
    save_settings,
)
from nl_db.llm.anthropic_provider import AnthropicProvider
from nl_db.llm.openai_compatible import OpenAICompatibleProvider
from nl_db.llm.openai_provider import OpenAIProvider
from nl_db.llm.provider import LLMProvider
from nl_db.pipeline import Pipeline, PipelineOutput
from nl_db.schema.base import render_for_prompt
from nl_db.schema.sqlite import SQLiteSchemaExtractor
from nl_db.validator import SQLValidationError


def _make_capturing_http_client(captures: list[dict[str, Any]]) -> httpx.Client:
    """Build an httpx client whose request event hook appends every outgoing
    request to `captures` so the UI can show the exact wire-level payload.
    """

    def _hook(request: httpx.Request) -> None:
        body: Any
        if request.content:
            try:
                body = json.loads(request.content)
            except Exception:  # noqa: BLE001
                body = request.content.decode("utf-8", errors="replace")
        else:
            body = None
        captures.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": {
                    k: (
                        "Bearer <redacted>"
                        if k.lower() == "authorization"
                        else ("<redacted>" if k.lower() == "x-api-key" else v)
                    )
                    for k, v in request.headers.items()
                },
                "body": body,
            }
        )

    def _response_hook(response: httpx.Response) -> None:
        # response.read() populates response.content; httpx caches it so the
        # caller (the openai/anthropic SDK) can still .json() it.
        try:
            response.read()
        except Exception:  # noqa: BLE001
            return
        body: Any
        try:
            body = json.loads(response.content)
        except Exception:  # noqa: BLE001
            body = response.content.decode("utf-8", errors="replace")
        if captures:
            # Attach to the most recently captured request — request hook
            # fires immediately before the network call, so the last entry
            # is the one this response belongs to.
            captures[-1]["response_status"] = response.status_code
            captures[-1]["response_body"] = body
            captures[-1]["response_text"] = _extract_response_text(body)

    return httpx.Client(
        event_hooks={"request": [_hook], "response": [_response_hook]},
        timeout=httpx.Timeout(60.0),
    )


def _extract_response_text(body: Any) -> str | None:
    """Pull the LLM's text content out of a parsed response body.

    Handles OpenAI chat-completions and Anthropic messages shapes. Returns
    None if the body doesn't match a known shape (e.g. error response).
    """
    if not isinstance(body, dict):
        return None
    # OpenAI: {"choices": [{"message": {"content": "..."}}]}
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
    # Anthropic: {"content": [{"type": "text", "text": "..."}, ...]}
    content_blocks = body.get("content")
    if isinstance(content_blocks, list):
        parts = [
            b.get("text", "")
            for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        if parts:
            return "".join(parts)
    return None


def _preview_outgoing_request(
    messages: list[Any],
    *,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Construct the HTTP request body that nl-db WOULD send to the configured
    provider, without actually sending it.

    Useful for diagnosing config problems against a broken endpoint — you can
    see the model name, headers, and base URL without any network call.
    """
    provider_name = st.session_state.provider_name
    model = st.session_state.model
    msgs_wire = [{"role": m.role, "content": m.content} for m in messages]

    if provider_name == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        system_parts = [m["content"] for m in msgs_wire if m["role"] == "system"]
        convo = [m for m in msgs_wire if m["role"] != "system"]
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        headers = {
            "x-api-key": "<redacted>",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    elif provider_name == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        body = {
            "model": model,
            "messages": msgs_wire,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        headers = {
            "Authorization": "Bearer <redacted>",
            "Content-Type": "application/json",
        }
    elif provider_name == "openai_compatible":
        base = (st.session_state.base_url or "").rstrip("/")
        url = f"{base}/chat/completions" if base else "(base_url not set)"
        body = {
            "model": model,
            "messages": msgs_wire,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        headers = {
            "Authorization": "Bearer <redacted>",
            "Content-Type": "application/json",
        }
    else:
        raise RuntimeError(f"Unknown provider: {provider_name}")

    return {"method": "POST", "url": url, "headers": headers, "body": body}


def _curl_equivalent(req: dict[str, Any]) -> str:
    """Render a captured request as a copy-pasteable curl command."""
    skip = {
        "host",
        "content-length",
        "user-agent",
        "accept-encoding",
        "connection",
        "accept",
    }
    lines = [f"curl -X {req['method']} '{req['url']}' \\"]
    for k, v in req["headers"].items():
        if k.lower() in skip:
            continue
        # show that auth was masked so the user knows to fill it in
        lines.append(f"  -H '{k}: {v}' \\")
    body = req["body"]
    if body is not None:
        body_str = (
            json.dumps(body, indent=2) if isinstance(body, (dict, list)) else str(body)
        )
        lines.append(f"  -d '{body_str}'")
    else:
        # drop trailing backslash
        lines[-1] = lines[-1].rstrip(" \\")
    return "\n".join(lines)


def _explain_llm_error(e: Exception) -> str:
    """Build a more useful error message for common LLM-call failures."""
    provider = st.session_state.provider_name
    model = st.session_state.model
    base = st.session_state.base_url or "(provider default)"
    where = (
        f"provider=**{provider}**, model=**`{model}`**, base_url=**`{base}`**"
    )

    if isinstance(e, (anthropic.NotFoundError, openai.NotFoundError)):
        hint = (
            "The LLM API returned **404 Not Found**. The most common cause is a "
            "model name that doesn't exist for this provider/account. Check the "
            "Model field in the sidebar against your provider's model list. "
            "For an OpenAI-compatible server, also verify the Base URL "
            "(e.g. Ollama is `http://localhost:11434/v1`, not 8080)."
        )
        return f"{hint}\n\nCalled with: {where}\n\nRaw: `{e}`"

    if isinstance(e, (anthropic.AuthenticationError, openai.AuthenticationError)):
        return (
            f"The LLM API rejected the API key. Re-check the **API key** field "
            f"in the sidebar.\n\nCalled with: {where}\n\nRaw: `{e}`"
        )

    if isinstance(e, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return (
            f"Couldn't reach the LLM endpoint. If you're using `openai_compatible`, "
            f"confirm your local server is running.\n\nCalled with: {where}\n\nRaw: `{e}`"
        )

    return f"{type(e).__name__}: {e}\n\nCalled with: {where}"


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
    g = base.generation
    st.session_state.initialized = True
    st.session_state.provider_name = base.provider.name
    st.session_state.model = base.provider.model
    st.session_state.base_url = base.provider.base_url or ""
    st.session_state.api_key = (
        base.provider.api_key.get_secret_value() if base.provider.api_key else ""
    )
    st.session_state.db_path = str(base.db.path) if base.db.path else ""
    st.session_state.max_rows = base.limits.max_rows
    st.session_state.timeout_s = base.limits.timeout_s
    st.session_state.max_prompt_tokens = base.limits.max_prompt_tokens
    st.session_state.temperature = g.temperature
    st.session_state.max_output_tokens = g.max_output_tokens
    st.session_state.paraphrase_enabled = g.paraphrase
    st.session_state.paraphrase_temperature = g.paraphrase_temperature
    st.session_state.paraphrase_max_output_tokens = g.paraphrase_max_output_tokens
    st.session_state.auto_limit = g.auto_limit
    st.session_state.num_few_shot = g.num_few_shot
    st.session_state.allow_writes = False
    st.session_state.history: list[HistoryEntry] = []
    st.session_state.last_output: PipelineOutput | None = None
    st.session_state.edited_sql: str | None = None
    st.session_state.last_captures: list[dict[str, Any]] = []
    st.session_state.preview_only: bool = False


def _snapshot_settings_from_state() -> Settings:
    """Build a Settings object from current sidebar state, ready to save.

    API key is deliberately NOT included — secrets stay in env / .env.
    """
    provider = ProviderConfig(
        name=st.session_state.provider_name,
        model=st.session_state.model,
        base_url=st.session_state.base_url or None,
    )
    limits = LimitsConfig(
        max_rows=int(st.session_state.max_rows),
        timeout_s=float(st.session_state.timeout_s),
        max_prompt_tokens=int(st.session_state.max_prompt_tokens),
    )
    generation = GenerationConfig(
        temperature=float(st.session_state.temperature),
        max_output_tokens=int(st.session_state.max_output_tokens),
        paraphrase=bool(st.session_state.paraphrase_enabled),
        paraphrase_temperature=float(st.session_state.paraphrase_temperature),
        paraphrase_max_output_tokens=int(st.session_state.paraphrase_max_output_tokens),
        auto_limit=bool(st.session_state.auto_limit),
        num_few_shot=int(st.session_state.num_few_shot),
    )
    snapshot = Settings.model_construct(
        provider=provider,
        db=Settings().db.model_copy(
            update={
                "path": Path(st.session_state.db_path).expanduser()
                if st.session_state.db_path
                else None
            }
        ),
        limits=limits,
        generation=generation,
    )
    return snapshot


_init_state()


# ---------------------------------------------------------------------------
# Provider/pipeline construction from session state
# ---------------------------------------------------------------------------

def _build_provider() -> tuple[LLMProvider, list[dict[str, Any]]]:
    """Build the configured provider plus a list that will capture every HTTP
    request the provider sends. Captures are appended in order.
    """
    name = st.session_state.provider_name
    model = st.session_state.model
    key = st.session_state.api_key
    captures: list[dict[str, Any]] = []
    http = _make_capturing_http_client(captures)

    if name == "anthropic":
        if not key:
            raise RuntimeError("Anthropic API key is required.")
        sdk = anthropic.Anthropic(api_key=key, http_client=http)
        return AnthropicProvider(model=model, api_key=key, client=sdk), captures
    if name == "openai":
        if not key:
            raise RuntimeError("OpenAI API key is required.")
        sdk = openai.OpenAI(api_key=key, http_client=http)
        return OpenAIProvider(model=model, api_key=key, client=sdk), captures
    if name == "openai_compatible":
        base_url = st.session_state.base_url
        if not base_url:
            raise RuntimeError("Base URL required for openai_compatible.")
        sdk = openai.OpenAI(
            api_key=key or "not-needed", base_url=base_url, http_client=http
        )
        return (
            OpenAICompatibleProvider(
                model=model, base_url=base_url, api_key=key, client=sdk
            ),
            captures,
        )
    raise RuntimeError(f"Unknown provider: {name}")


def _build_pipeline() -> tuple[Pipeline, list[dict[str, Any]]]:
    db_path = Path(st.session_state.db_path).expanduser()
    if not db_path.exists():
        raise RuntimeError(f"Database not found: {db_path}")
    n_few_shot: int | None = (
        None if st.session_state.num_few_shot == -1 else st.session_state.num_few_shot
    )
    provider, captures = _build_provider()
    pipeline = Pipeline(
        provider=provider,
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
    return pipeline, captures


# ---------------------------------------------------------------------------
# Sidebar — all the knobs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("nl-db")
    cfg_path = default_config_path()
    st.caption(
        f"Edits are session-scoped. Click **Save to disk** to persist to "
        f"`{cfg_path}` (API key never written)."
    )

    save_col, reload_col = st.columns(2)
    with save_col:
        if st.button("💾 Save to disk", use_container_width=True):
            try:
                written = save_settings(_snapshot_settings_from_state(), cfg_path)
                st.success(f"Wrote {written}")
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")
    with reload_col:
        if st.button("↻ Reload from disk", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

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
        # Always render the sub-widgets so Streamlit doesn't clear their keys
        # from session_state when paraphrase is toggled off. Disable when off
        # so they look obviously inactive.
        _para_off = not st.session_state.paraphrase_enabled
        st.slider(
            "Temperature (paraphrase)",
            0.0,
            1.5,
            key="paraphrase_temperature",
            step=0.05,
            disabled=_para_off,
        )
        st.number_input(
            "Max output tokens (paraphrase)",
            min_value=32,
            max_value=512,
            step=16,
            key="paraphrase_max_output_tokens",
            disabled=_para_off,
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

tab_query, tab_chat, tab_schema, tab_history, tab_about = st.tabs(
    ["Query", "Chat", "Schema", "History", "About"]
)


# --- Query tab -------------------------------------------------------------

with tab_query:
    if not st.session_state.db_path:
        st.info("Set a SQLite path in the sidebar to get started.", icon="📁")
    else:
        col_q, col_gen, col_preview = st.columns([5, 1, 1], vertical_alignment="bottom")
        with col_q:
            question = st.text_area(
                "Question",
                key="question_input",
                placeholder='"How much did each user spend last month?"',
                height=80,
            )
        with col_gen:
            ask_clicked = st.button(
                "Ask", type="primary", use_container_width=True
            )
        with col_preview:
            preview_clicked = st.button(
                "Preview only",
                use_container_width=True,
                help=(
                    "Builds the prompt and shows the exact HTTP request body "
                    "nl-db would send — no LLM call is made. Useful when your "
                    "endpoint is misconfigured and Ask fails."
                ),
            )

        # Reserve a visual slot for the Debug expander right under the buttons.
        # We write into it AT THE END of the tab block — after both Preview
        # and Generate SQL handlers have populated session_state.last_captures.
        # Using a container (vs. rendering inline) decouples render position
        # from render time, so a captures-list populated later in the same
        # script run still shows up high in the layout.
        debug_slot = st.container()

        if preview_clicked and question.strip():
            try:
                from nl_db.prompts.builder import build_sql_prompt

                # Build the prompt without needing API keys or a working endpoint.
                # We need the schema to inject — that only requires reading the SQLite file.
                from nl_db.schema.cache import SchemaCache
                from nl_db.schema.sqlite import SQLiteSchemaExtractor

                db_path = Path(st.session_state.db_path).expanduser()
                if not db_path.exists():
                    raise RuntimeError(f"Database not found: {db_path}")
                schema = SchemaCache().get(
                    db_path,
                    lambda: SQLiteSchemaExtractor.from_path(db_path).extract(),
                )
                n_few_shot: int | None = (
                    None
                    if st.session_state.num_few_shot == -1
                    else st.session_state.num_few_shot
                )
                if n_few_shot is None:
                    examples = None
                elif n_few_shot <= 0:
                    examples = ()
                else:
                    from nl_db.prompts.examples import few_shot_for

                    examples = few_shot_for(schema.dialect)[:n_few_shot]

                prompt = build_sql_prompt(schema, question, examples=examples)
                preview_req = _preview_outgoing_request(
                    prompt.messages,
                    temperature=float(st.session_state.temperature),
                    max_output_tokens=int(st.session_state.max_output_tokens),
                )
                st.session_state.last_captures = [preview_req]
                st.session_state.preview_only = True
                # Clear any prior real-call output so the page reflects "preview only" state.
                st.session_state.last_output = None
                st.session_state.edited_sql = None
                st.success(
                    "Built the request body locally — no network call was made.",
                    icon="📡",
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}", icon="❌")

        if ask_clicked and question.strip():
            st.session_state.last_captures = []
            st.session_state.preview_only = False
            with st.spinner("Asking the database…"):
                try:
                    from nl_db.generator import Answer, CannotAnswer, Clarify

                    pipeline, captures = _build_pipeline()
                    st.session_state.last_captures = captures
                    # confirm=lambda True: run the full pipeline including
                    # execution. Read-only requests are inherently safe;
                    # destructive SQL is already blocked by the validator
                    # unless Allow writes is checked (an explicit opt-in).
                    pout = pipeline.run(
                        question,
                        allow_writes=bool(st.session_state.allow_writes),
                        confirm=lambda _sql, _para: True,
                    )
                    if isinstance(pout.outcome, Answer):
                        assert pout.sql_final is not None
                        assert pout.result is not None
                        st.session_state.last_output = {
                            "kind": "answer",
                            "question": question,
                            "sql_final": pout.sql_final,
                            "paraphrase": pout.paraphrase,
                            "is_destructive": pout.is_destructive,
                            "auto_limit_applied": pout.auto_limit_applied,
                            "approx_prompt_tokens": pout.prompt.approx_tokens,
                            "columns": list(pout.result.columns),
                            "rows": [list(r) for r in pout.result.rows],
                            "row_count": pout.result.row_count,
                            "truncated": pout.result.truncated,
                        }
                        st.session_state.history.insert(
                            0,
                            HistoryEntry(
                                ts=time.time(),
                                question=question,
                                sql=pout.sql_final,
                                paraphrase=pout.paraphrase,
                                row_count=pout.result.row_count,
                                success=True,
                            ),
                        )
                    elif isinstance(pout.outcome, CannotAnswer):
                        st.session_state.last_output = {
                            "kind": "cannot_answer",
                            "question": question,
                            "reason": pout.outcome.reason,
                            "available_tables": list(pout.outcome.available_tables),
                            "approx_prompt_tokens": pout.prompt.approx_tokens,
                        }
                    elif isinstance(pout.outcome, Clarify):
                        st.session_state.last_output = {
                            "kind": "clarify",
                            "question": question,
                            "clarify_question": pout.outcome.question,
                            "approx_prompt_tokens": pout.prompt.approx_tokens,
                        }
                except SQLValidationError as e:
                    st.error(_explain_llm_error(e), icon="🛑")
                    st.session_state.last_output = None
                except Exception as e:  # noqa: BLE001
                    st.error(_explain_llm_error(e), icon="❌")
                    st.session_state.last_output = None

        out = st.session_state.last_output
        if out and out.get("kind") == "cannot_answer":
            st.info(out["reason"], icon="🤷")
            tables = out["available_tables"] or ["(none)"]
            st.caption("**Available tables in this database:** " + ", ".join(tables))
        elif out and out.get("kind") == "clarify":
            st.warning(out["clarify_question"], icon="❓")
            clarification = st.text_input(
                "Your answer", key="clarify_response_input"
            )
            if st.button("Re-ask with clarification", type="primary") and clarification.strip():
                st.session_state.question_input = (
                    f"{out['question']}\n\nClarification: {clarification}"
                )
                st.session_state.last_output = None
                st.session_state.last_captures = []
                st.rerun()
        elif out and out.get("kind") == "answer":
            # Result-first: dataframe + meta row at the top.
            if out["columns"]:
                df = pd.DataFrame(out["rows"], columns=out["columns"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Query produced no columns.")

            meta_a, meta_b, meta_c = st.columns(3)
            meta_a.metric("Rows", out["row_count"])
            meta_b.metric(
                "Truncated", "yes" if out["truncated"] else "no"
            )
            meta_c.metric(
                "Auto-LIMIT",
                "applied" if out["auto_limit_applied"] else "skipped",
            )

            # SQL + paraphrase live in a collapsible section below — the user
            # can verify what we actually ran (SQL transparency invariant)
            # without having SQL be the primary visual focus.
            with st.expander("How I answered (SQL + paraphrase)", expanded=False):
                if out["paraphrase"]:
                    st.success(out["paraphrase"], icon="🗣️")
                st.code(out["sql_final"], language="sql")
                st.caption(
                    f"~{out['approx_prompt_tokens']} prompt tokens · "
                    f"{'destructive' if out['is_destructive'] else 'read-only'}"
                )

            with st.expander("Raw JSON"):
                st.code(
                    json.dumps(
                        {
                            "columns": out["columns"],
                            "rows": out["rows"],
                            "row_count": out["row_count"],
                            "truncated": out["truncated"],
                        },
                        default=str,
                        indent=2,
                    ),
                    language="json",
                )

    # Render Debug expander INTO the slot we reserved up high (right under
    # the buttons). Writing here, at the very end of the tab, guarantees the
    # session_state.last_captures has already been populated by whichever
    # handler ran above (Preview only or Ask).
    with debug_slot:
        captures: list[dict[str, Any]] = st.session_state.get(
            "last_captures", []
        )
        if captures:
            is_preview = bool(st.session_state.get("preview_only", False))
            label = (
                f"🐞 Preview: {len(captures)} request body (no call made)"
                if is_preview
                else f"🐞 Debug: {len(captures)} LLM API call(s) — request + response"
            )
            with st.expander(label, expanded=True):
                if is_preview:
                    st.caption(
                        "This is the exact JSON nl-db would POST. Verify "
                        "`model`, headers, and URL match what your server expects."
                    )
                call_labels = [
                    "SQL generation" if i == 0 else "Paraphrase"
                    for i in range(len(captures))
                ]
                for i, (lab, req) in enumerate(
                    zip(call_labels, captures, strict=False)
                ):
                    st.markdown(f"**Call {i + 1}: {lab}**")
                    status_suffix = (
                        f"  →  HTTP {req['response_status']}"
                        if "response_status" in req
                        else ""
                    )
                    st.code(
                        f"{req['method']} {req['url']}{status_suffix}",
                        language="http",
                    )

                    has_response = "response_body" in req
                    tab_names = ["Request body", "Headers", "curl"]
                    if has_response:
                        tab_names = [
                            "Response text",
                            "Response (full)",
                            "Request body",
                            "Headers",
                            "curl",
                        ]
                    tabs = st.tabs(tab_names)

                    if has_response:
                        with tabs[0]:
                            text = req.get("response_text")
                            if text:
                                st.caption(
                                    "What the LLM actually said. If you're "
                                    "seeing 'no SQL' errors, verify here."
                                )
                                st.code(text, language="markdown")
                            else:
                                st.warning(
                                    "The response had no extractable text "
                                    "field (e.g. `choices[0].message.content` "
                                    "was empty). Almost always means the shim "
                                    "or model emitted nothing — increase "
                                    "max_output_tokens, check the shim's "
                                    "non-streaming path, or try a different "
                                    "model."
                                )
                                st.json(
                                    req["response_body"]
                                    if isinstance(req["response_body"], (dict, list))
                                    else {"raw": str(req["response_body"])}
                                )
                        with tabs[1]:
                            if isinstance(req["response_body"], (dict, list)):
                                st.code(
                                    json.dumps(req["response_body"], indent=2),
                                    language="json",
                                )
                            else:
                                st.code(str(req["response_body"]))

                    offset = 2 if has_response else 0
                    with tabs[offset]:
                        if isinstance(req["body"], (dict, list)):
                            st.code(
                                json.dumps(req["body"], indent=2),
                                language="json",
                            )
                        else:
                            st.code(str(req["body"]))
                    with tabs[offset + 1]:
                        st.code(
                            "\n".join(
                                f"{k}: {v}" for k, v in req["headers"].items()
                            )
                        )
                    with tabs[offset + 2]:
                        st.caption(
                            "Authorization is redacted — fill in your key before running."
                        )
                        st.code(_curl_equivalent(req), language="bash")
                    if i < len(captures) - 1:
                        st.divider()


# --- Chat tab --------------------------------------------------------------

with tab_chat:
    if not st.session_state.db_path:
        st.info("Set a SQLite path in the sidebar to start a chat.", icon="📁")
    else:
        if "chat_conversation" not in st.session_state:
            from nl_db.conversation import Conversation

            st.session_state.chat_conversation = Conversation()

        st.caption(
            "Multi-turn chat — the conversation history is passed back into "
            "each prompt, so follow-ups like *'now group by region'* work."
        )

        col_reset, _ = st.columns([1, 5])
        with col_reset:
            if st.button("🗑️ Clear chat", use_container_width=True):
                from nl_db.conversation import Conversation

                st.session_state.chat_conversation = Conversation()
                st.rerun()

        # Render transcript
        from nl_db.generator import Answer, CannotAnswer, Clarify

        for turn in st.session_state.chat_conversation.turns:
            with st.chat_message("user"):
                st.write(turn.question)
            with st.chat_message("assistant"):
                if isinstance(turn.outcome, Answer):
                    st.code(turn.outcome.sql, language="sql")
                    if turn.row_summary:
                        st.caption(turn.row_summary)
                elif isinstance(turn.outcome, CannotAnswer):
                    st.info(turn.outcome.reason, icon="🤷")
                elif isinstance(turn.outcome, Clarify):
                    st.warning(turn.outcome.question, icon="❓")

        chat_question = st.chat_input("Ask a follow-up…")
        if chat_question:
            from nl_db.conversation import Turn, summarize_rows

            try:
                pipeline, _captures = _build_pipeline()
                pout = pipeline.run(
                    chat_question,
                    allow_writes=False,
                    confirm=lambda _sql, _para: True,  # auto-execute in chat
                    history=st.session_state.chat_conversation,
                )
                row_summary: str | None = None
                if isinstance(pout.outcome, Answer) and pout.result is not None:
                    row_summary = summarize_rows(
                        pout.result.columns, pout.result.rows
                    )
                st.session_state.chat_conversation.append(
                    Turn(
                        question=chat_question,
                        outcome=pout.outcome,
                        row_summary=row_summary,
                    )
                )
                # Stash the latest result rows for inline display below the
                # transcript on the next rerun.
                if isinstance(pout.outcome, Answer) and pout.result is not None:
                    st.session_state.chat_last_rows = {
                        "columns": list(pout.result.columns),
                        "rows": [list(r) for r in pout.result.rows],
                        "truncated": pout.result.truncated,
                    }
                else:
                    st.session_state.chat_last_rows = None
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(_explain_llm_error(e), icon="❌")

        last_rows = st.session_state.get("chat_last_rows")
        if last_rows:
            st.subheader("Latest result")
            df = pd.DataFrame(last_rows["rows"], columns=last_rows["columns"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            if last_rows["truncated"]:
                st.caption("Result was truncated by max_rows.")


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
