# nl-db

Natural language database querying. Privacy-first by design (on-device via Apple Intelligence), provider-agnostic by construction (any OpenAI-compatible endpoint, plus Anthropic and OpenAI directly), and exposable as an MCP server to any compatible client.

## Quickstart

```bash
uv sync

# Configure your provider (any one of these is enough)
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env      # Claude
# OR
echo "OPENAI_API_KEY=sk-..." >> .env             # GPT
# OR (for Ollama / Apple Intelligence shim / vLLM)
export NL_DB_PROVIDER__NAME=openai_compatible
export NL_DB_PROVIDER__BASE_URL=http://localhost:8080/v1
export NL_DB_PROVIDER__MODEL=apple-intelligence

# Inspect the schema
uv run nl-db schema --db path/to/your.db

# Ask a question
uv run nl-db query "list the 5 most recent transactions" --db path/to/your.db

# Non-interactive JSON
uv run nl-db query "total spend by category" --db path/to/your.db --json --no-confirm
```

Every query:

1. Inspects the live schema (cached by file mtime)
2. Generates SQL via your configured LLM
3. Validates with `sqlglot` and auto-injects `LIMIT` on unbounded reads
4. Shows you the SQL **plus** a one-sentence plain-English paraphrase
5. Waits for your confirmation (skip with `--no-confirm`)
6. Executes and prints a table (or JSON with `--json`)
7. Logs the run as JSONL to `~/.local/share/nl-db/logs/`

Writes (`INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`TRUNCATE`) are refused unless you pass `--allow-writes`.

## Streamlit playground

A session-scoped UI for experimentation — edit every config knob (provider, model, key, DB path, temperature, max tokens, paraphrase on/off, auto-LIMIT, few-shot count, …), inspect the schema, run NL queries, edit the generated SQL, and see results in a dataframe.

```bash
uv run nl-db-ui
# opens http://localhost:8501
```

Nothing typed in the UI is written to disk — config edits are session-scoped. To persist, use `~/.config/nl-db/config.toml` or `.env`.

## MCP server

`nl-db` also runs as an MCP stdio server, exposing four tools and a schema resource to any compatible client (Claude Desktop, Cursor, …):

| Tool | Purpose |
| --- | --- |
| `list_tables` | Every user-created table in the DB |
| `describe_schema(table_name)` | Columns, types, PKs, FKs for one table |
| `query_database(question)` | Full NL → SQL → result pipeline |
| `run_sql(sql)` | Execute raw SQL (gated by `--allow-writes`) |

Resource: `db://schema/<table>` returns the same payload as `describe_schema`.

Register with Claude Desktop by adding to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nl-db": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/nl-db", "run", "nl-db-mcp", "--db", "/absolute/path/to/your.db"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

Full setup walkthrough in [`docs/setup.md`](docs/setup.md).

## Configuration

Precedence: env vars > `.env` > `./nl-db.toml` (project-local) > defaults.

The config file is **project-local** — it lives next to your code at `./nl-db.toml`. Override the location via `NL_DB_CONFIG_FILE=path/to/file.toml`.

```toml
# ./nl-db.toml
[provider]
name = "anthropic"                 # "anthropic" | "openai" | "openai_compatible"
model = "claude-sonnet-4-6"
# base_url = "http://localhost:8080/v1"   # required if name = "openai_compatible"

[limits]
max_rows = 1000
timeout_s = 10.0
max_prompt_tokens = 8000

[generation]
temperature = 0.0
max_output_tokens = 512
paraphrase = true
paraphrase_temperature = 0.0
paraphrase_max_output_tokens = 128
auto_limit = true
num_few_shot = -1                  # -1 = all curated examples; 0 = none
```

The Streamlit UI has a **Save to disk** button in the sidebar that writes the current settings to `./nl-db.toml`. API keys are never written — secrets always stay in `.env` or env vars. `nl-db.toml` is gitignored by default (uncomment in `.gitignore` to share team config).

API keys come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENAI_COMPATIBLE_API_KEY` — never write them into `config.toml`.

## Design

See [`plan.md`](plan.md) for the full architecture and the `/Users/shreyansjain/.claude/plans/...` build plan.

Key invariants:
- **Schema-first prompting**: live schema is always injected — no stale snapshots.
- **SQL transparency**: generated SQL is shown to the user (or the host LLM, via MCP) before execution.
- **Read-only by default**: writes require an explicit flag.
- **Provider-agnostic**: every LLM call goes through `LLMProvider` Protocol. Adding a new backend is one file.
- **Eval-driven**: 30 NL→SQL pairs (`eval/dataset.yaml`) score every change.

## Development

```bash
uv sync
uv run pytest                          # 72 tests
uv run ruff check
uv run mypy src/

# Build the sample database (gitignored, regenerable)
uv run python tests/fixtures/build_sample_db.py

# Run the eval harness
uv run python -m eval.runner --limit 5
```

## Status

- [x] Config layer (env / .env / TOML precedence)
- [x] Schema extractor + cache (SQLite)
- [x] LLM provider abstraction (Anthropic, OpenAI, OpenAI-compatible)
- [x] Prompt builder + few-shot + paraphrase
- [x] SQL generator + sqlglot validator + auto-LIMIT
- [x] Query executor + table/JSON formatters
- [x] Pipeline orchestrator
- [x] CLI (`nl-db query`, `nl-db schema`, `nl-db config`)
- [x] Sample database fixture
- [x] 30-question eval harness
- [x] MCP stdio server (4 tools + schema resource)
- [x] Streamlit playground UI (`nl-db-ui`)
- [x] Setup docs

### Deferred to post-v1

- PostgreSQL + MySQL adapters (drop into the existing `SchemaExtractor` Protocol slot)
- NL result summarizer (`--summarize`)
- Multi-connection config manager
- Apple Intelligence Swift HTTP shim (provided by a third-party project; nl-db is already compatible via the `openai_compatible` provider)
