# NLQ — Natural Language Queries

Ask your database questions in plain English — privately, locally, and from anywhere. NLQ runs on top of any LLM you pick (Apple Intelligence on-device, Ollama, vLLM, Anthropic, or OpenAI), is read-only by default, and plugs straight into Claude Desktop / Cursor as an MCP server.

> The CLI / Python package is named `nl-db` (predates the NLQ rename); the project as a product is **NLQ**.

## Quickstart

```bash
uv sync

# Configure your provider (any one of these is enough)
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env      # Claude
# OR
echo "OPENAI_API_KEY=sk-..." >> .env             # GPT
# OR (for Ollama / Apple Intelligence shim / vLLM)
export NL_DB_PROVIDER__NAME=openai_compatible
export NL_DB_PROVIDER__BASE_URL=http://localhost:11434/v1
export NL_DB_PROVIDER__MODEL=gemma2:2b

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

If the database can't answer a question, NLQ says so plainly (`CANNOT_ANSWER`) instead of hallucinating SQL. Ambiguous questions trigger a `CLARIFY` follow-up so the LLM can ask you a question back.

## Streamlit playground

A web UI for experimentation — edit every config knob (provider, model, key, DB path, temperature, max tokens, paraphrase on/off, auto-LIMIT, few-shot count, …), inspect the schema, ask questions, see results inline, and dig into the raw HTTP request/response with the debug toggle.

```bash
uv run nl-db-ui
# opens http://localhost:8501
```

Sidebar settings are session-scoped until you click **💾 Save to disk**, which persists them to `./nl-db.toml`. API keys are never written — secrets always stay in `.env` or env vars.

## Use NLQ with Claude Desktop

NLQ exposes itself as an [MCP](https://modelcontextprotocol.io) stdio server. Once registered with Claude Desktop, Claude can ask your database questions in natural language without ever writing SQL itself.

### 1. Locate Claude Desktop's config file

```bash
# macOS
~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
%APPDATA%\Claude\claude_desktop_config.json
```

If the file doesn't exist yet, create it with `{}` as the contents.

### 2. Add the NLQ entry

Pick the snippet that matches the LLM provider you want Claude to use behind the scenes:

#### Anthropic (Claude as the underlying NL→SQL model)

```json
{
  "mcpServers": {
    "nlq": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/nl-db",
        "run", "nl-db-mcp",
        "--db", "/absolute/path/to/your.db"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "NL_DB_PROVIDER__NAME": "anthropic",
        "NL_DB_PROVIDER__MODEL": "claude-sonnet-4-5-20250929"
      }
    }
  }
}
```

#### OpenAI

```json
{
  "mcpServers": {
    "nlq": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/nl-db",
        "run", "nl-db-mcp",
        "--db", "/absolute/path/to/your.db"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "NL_DB_PROVIDER__NAME": "openai",
        "NL_DB_PROVIDER__MODEL": "gpt-4o-mini"
      }
    }
  }
}
```

#### Local / OpenAI-compatible (Ollama, Apple Intelligence shim, vLLM, LM Studio)

```json
{
  "mcpServers": {
    "nlq": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/nl-db",
        "run", "nl-db-mcp",
        "--db", "/absolute/path/to/your.db"
      ],
      "env": {
        "NL_DB_PROVIDER__NAME": "openai_compatible",
        "NL_DB_PROVIDER__BASE_URL": "http://localhost:11434/v1",
        "NL_DB_PROVIDER__MODEL": "gemma2:2b",
        "OPENAI_COMPATIBLE_API_KEY": "not-needed"
      }
    }
  }
}
```

If you already have an `nl-db.toml` set up locally (e.g. via the Streamlit playground's **Save to disk** button), simplify the `env` block to:

```json
"env": {
  "NL_DB_CONFIG_FILE": "/absolute/path/to/nl-db/nl-db.toml",
  "OPENAI_COMPATIBLE_API_KEY": "not-needed"
}
```

### 3. Restart Claude Desktop

Fully quit (`Cmd+Q` on macOS) and relaunch. Open a new chat — you should see the 🔌 plug icon show `nlq` connected with four tools available.

### 4. Try it

> *"What tables are in the database?"* — Claude calls `list_tables`.
>
> *"How much did Alice spend last month?"* — Claude calls `describe_database` to ground itself in the schema, then `query_database(question)` and returns the answer in plain English.

NLQ returns one of three response shapes via `query_database`, so Claude knows whether it got an answer, a refusal (with the list of available tables), or a clarifying question to ask you back.

### Power-user flag

Add `"--expose-run-sql"` to `args` to register an extra `run_sql(sql)` tool that lets Claude execute SQL directly (bypassing the NL pipeline). Add `"--allow-writes"` alongside it to permit destructive statements. Both are off by default.

If something goes wrong, Claude Desktop logs the MCP server's stderr at `~/Library/Logs/Claude/mcp-server-nlq.log`. The full setup walkthrough lives in [`docs/setup.md`](docs/setup.md).

## MCP tool surface

| Tool | Purpose |
| --- | --- |
| `list_tables()` | Every user-created table in the DB |
| `describe_database()` | Full schema (every table, columns, FKs) in one call |
| `describe_schema(table_name)` | Schema for one specific table |
| `query_database(question, conversation_id?)` | Full NL → SQL → answer pipeline; supports multi-turn via `conversation_id` |
| `run_sql(sql)` | Execute raw SQL (only when `--expose-run-sql` is set) |

Resources: `db://schema` (full schema) and `db://schema/<table>` (per-table).

## Configuration

Precedence: env vars > `.env` > `./nl-db.toml` (project-local) > defaults.

```toml
# ./nl-db.toml
[provider]
name = "openai_compatible"          # "anthropic" | "openai" | "openai_compatible"
model = "gemma2:2b"
base_url = "http://localhost:11434/v1"   # required if name = "openai_compatible"

[limits]
max_rows = 1000
timeout_s = 10.0
max_prompt_tokens = 8000

[generation]
temperature = 0.0
max_output_tokens = 2048
paraphrase = true
paraphrase_temperature = 0.0
paraphrase_max_output_tokens = 512
auto_limit = true
num_few_shot = -1                   # -1 = all curated examples; 0 = none
```

The Streamlit UI's sidebar **💾 Save to disk** button writes this file for you. API keys are never written — they always come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENAI_COMPATIBLE_API_KEY` (env or `.env`).

## Design

Key invariants:
- **Schema-first prompting**: live schema is always injected — no stale snapshots.
- **Three-state generator output**: every NL question resolves to `ANSWER` (with SQL), `CANNOT_ANSWER` (with reason + available tables), or `CLARIFY` (with a follow-up question). No silent hallucination.
- **NL-friendly errors**: raw exceptions are humanized before reaching the user.
- **SQL transparency**: generated SQL is shown to the user (or the host LLM, via MCP) before/alongside execution.
- **Read-only by default**: writes require an explicit flag.
- **Provider-agnostic**: every LLM call goes through `LLMProvider` Protocol. Adding a new backend is one file.
- **Eval-driven**: 35 NL→SQL/state pairs (`eval/dataset.yaml`) score every change.

## Development

```bash
uv sync
uv run pytest                          # 132 tests, <1s
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
- [x] 35-question eval harness with three-state coverage
- [x] MCP stdio server (4 default tools + 1 opt-in + 2 schema resources)
- [x] Streamlit playground UI (`nl-db-ui`) with debug toggle
- [x] Setup docs
- [x] Three-state generator output (`ANSWER` / `CANNOT_ANSWER` / `CLARIFY`)
- [x] NL-first MCP surface (`describe_database`, top-level `db://schema` resource, `run_sql` behind `--expose-run-sql`)
- [x] Multi-turn chat (Streamlit Chat tab + MCP `conversation_id`)

### Deferred to post-v1

- PostgreSQL + MySQL adapters (drop into the existing `SchemaExtractor` Protocol slot)
- NL result summarizer (`--summarize`)
- Multi-connection config manager
- Apple Intelligence Swift HTTP shim (provided by a third-party project; NLQ is already compatible via the `openai_compatible` provider)
