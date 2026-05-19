# nl-db — plan

## Vision

nl-db lets anyone query a database in plain English. The audience is both:

- **End users** (often non-technical) via a chat UI — natural language in, answer out, with clarifying follow-ups when the question is ambiguous.
- **Host LLMs** via an MCP server — host LLMs send NL, not SQL. nl-db does the translation behind the scenes.

Both surfaces wrap the same pipeline. SQL is an implementation detail, not an interface. The CLI and the Streamlit playground keep SQL editable for power users; everywhere else the user / host LLM never has to think about SQL.

## Architecture

```
[Chat UI: end users]              [MCP server: host LLMs]
            ↓                                  ↓
        [Chat session — multi-turn, with clarifying follow-ups]
                                 ↓
                  [NL → SQL → answer pipeline]
                  - schema extraction (cached)
                  - prompt building (schema-first)
                  - LLM call (provider-agnostic)
                  - SQL validation (sqlglot; read-only default)
                  - query execution
                  - paraphrase + NL-error wrapping
                                 ↓
                          [Database]
```

## Core invariants (do not break)

- **Schema-first prompting.** Live schema is injected into every LLM call.
- **Three-state generator output.** Every NL question resolves to exactly one of:
  - `ANSWER(sql)` — the happy path
  - `CANNOT_ANSWER(reason, available_tables)` — schema doesn't contain the data; reason is plain English
  - `CLARIFY(question)` — request is ambiguous; we ask back in the user's vocabulary
- **NL-friendly errors.** No raw stack traces, sqlglot parse errors, or SQLite messages reach the user — they're translated to plain English first.
- **SQL transparency.** Generated SQL is surfaced alongside results (or in the MCP response) for review. The user / host LLM is never *asked* to write it.
- **Read-only by default.** Writes only with explicit `--allow-writes`. The MCP server hides `run_sql` behind an opt-in `--expose-run-sql` flag — the NL path covers the common case.
- **Provider-agnostic.** Every LLM call goes through the `LLMProvider` Protocol. Adding a new backend is one file.
- **Eval-driven.** Prompt changes regress only if eval catches them — so every new prompt state needs eval coverage.

## Roadmap (next iteration, ordered)

1. ✅ **Three-state generator** — sentinel-based prompt format yielding `ANSWER | CANNOT_ANSWER | CLARIFY`. NL-error surface module (`nl_db.nl_errors.humanize()`) that wraps raw exceptions before they reach any user-facing layer. *(done — `GenerationOutcome` + Pipeline rework)*
2. ✅ **MCP schema improvements + surface narrowing** — `describe_database()` tool + top-level `db://schema` Resource (one call grounds the host LLM in the full schema). `query_database` returns three states (`ANSWER` / `CANNOT_ANSWER` / `CLARIFY`). `run_sql` moved behind `--expose-run-sql` (default off); `--allow-writes` now requires `--expose-run-sql`. Tool descriptions rewritten as NL-first product copy.
3. **`CANNOT_ANSWER` carries hints** — `available_tables` (done — injected by Pipeline from live schema and surfaced in MCP responses) plus an optional `suggested_rephrase` (second LLM call). *Rephrase deferred pending eval data on cannot-answer frequency.*
4. ✅ **MCP surface narrows** — folded into #2 above.
5. ✅ **Eval cases for the new states** — 3 `CANNOT_ANSWER` + 2 `CLARIFY` cases added; runner does state-match before row/SQL scoring; per-case Markdown report shows actual vs expected state.
6. **UI three-state rendering + chat tab** — info banner for `CANNOT_ANSWER`, follow-up input for `CLARIFY`. New "Chat" tab in the Streamlit playground using the three-state output as the natural multi-turn trigger.
7. **Multi-turn chat** — conversation state in the pipeline; MCP `query_database` gains an optional `conversation_id` so host LLMs carry context across calls.

## Future work (post-roadmap, rough priority)

### Caching

- **Provider-side first.** Anthropic prompt caching (`cache_control: ephemeral` on the schema block) — ~30 mins of work, 30–50% latency win on repeat calls. OpenAI prompt caching is automatic when the prompt prefix is stable.
- **In-process LRU** for NL→SQL and paraphrase, keyed on `(schema_hash, question, model, temperature, num_few_shot, prompt_version)`. Lives in the MCP server process; the CLI doesn't need it.
- **Cache disabled in eval and tests by default** — caching hides regressions.
- **Cache observability** — hit/miss recorded in the existing JSONL log so we can measure hit rate before declaring success.
- **NOT caching query results.** The underlying DB may be written to by other processes; freshness is part of the contract. If ever added, behind an explicit TTL flag with a documented staleness window.

### Additional database backends

- **Postgres** and **MySQL** adapters drop into the existing `SchemaExtractor` Protocol slot and the `QueryExecutor` dispatch. Each is one new file + dialect-specific prompt tuning + new eval cases.
- **Dialect-aware system prompts** — currently SQLite-tuned. Postgres / MySQL each get their own.

### Schema enrichment (opt-in)

- `describe_database(include_stats=true)` returning row counts per table, date ranges on date columns, sample values for low-cardinality string columns.
- Opt-in because sample values risk leaking PII through the MCP boundary; row counts / date ranges cost extra queries on every fetch.

### Result summarizer

- A `--summarize` flag (CLI) / `summarize=true` parameter (MCP) that adds a second LLM pass over the result rows: *"The top three users by spend are Alice ($X), Bob ($Y), Carol ($Z)."* Distinct from the paraphrase (which describes the SQL).

### Observability and cost

- **Per-request cost estimate** in the JSONL log (provider × input_tokens × output_tokens × per-model rates).
- **Aggregate dashboard** (CLI subcommand) over the log — *"queries this week, hit rate, avg latency, total cost."*
- **Cross-provider eval comparison** — run the eval set against multiple configured providers and produce a diff report.

### Ergonomics

- **Web UI hardening** — single-password auth before exposing the Streamlit UI on anything other than localhost.
- **MCP HTTP / SSE transport** — currently stdio only. HTTP unlocks remote MCP clients.
- **Per-conversation MCP sessions** — if / when the server becomes a long-lived multi-tenant service.
