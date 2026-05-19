# nl-db setup guide

This guide covers:

1. Installation
2. Configuring an LLM provider (Anthropic, OpenAI, OpenAI-compatible — including Apple Intelligence)
3. Running the CLI
4. Registering nl-db as an MCP server with Claude Desktop and Cursor
5. Running the eval harness

## 1. Install

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone <repo> nl-db && cd nl-db
uv sync
```

## 2. Configure a provider

You can pick any **one** of: Anthropic (Claude), OpenAI (GPT), or any OpenAI-compatible endpoint (Apple Intelligence shim, Ollama, vLLM, LM Studio).

### A. Anthropic (Claude)

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

Optionally set the model (default: `claude-sonnet-4-6`):

```bash
export NL_DB_PROVIDER__MODEL=claude-opus-4-7
```

### B. OpenAI (GPT)

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
export NL_DB_PROVIDER__NAME=openai
export NL_DB_PROVIDER__MODEL=gpt-4o
```

### C. OpenAI-compatible (Apple Intelligence, Ollama, vLLM, …)

This is the path for any local or self-hosted server that speaks the OpenAI `/v1/chat/completions` wire format.

```bash
export NL_DB_PROVIDER__NAME=openai_compatible
export NL_DB_PROVIDER__BASE_URL=http://localhost:8080/v1   # your server
export NL_DB_PROVIDER__MODEL=apple-intelligence            # whatever your server expects
# many local servers don't require a real key; placeholder is fine:
echo "OPENAI_COMPATIBLE_API_KEY=not-needed" >> .env
```

**Apple Intelligence specifically:** nl-db does not bundle a server. Point `BASE_URL` at any third-party HTTP shim that exposes Apple's on-device `FoundationModels` framework through the OpenAI chat-completions format. Once the shim is running locally, nl-db treats it like any other OpenAI-compatible endpoint.

### Persisting choices

Anything you'd otherwise set in env vars can live in `./nl-db.toml` (next to your project):

```toml
[provider]
name = "openai_compatible"
model = "apple.local"
base_url = "http://localhost:8080/v1"

[limits]
max_rows = 500
timeout_s = 8.0

[generation]
temperature = 0.0
paraphrase = true
auto_limit = true
num_few_shot = -1
```

Move it elsewhere with `NL_DB_CONFIG_FILE=/path/to/other.toml`. The Streamlit UI has a **Save to disk** button that writes this file for you.

Env vars override TOML; TOML overrides built-in defaults.

Verify with `uv run nl-db config`.

## 3. CLI

```bash
# Inspect the schema
uv run nl-db schema --db tests/fixtures/sample.db

# Ask a question (interactive confirmation)
uv run nl-db query "total spend per category, highest first" \
    --db tests/fixtures/sample.db

# Skip confirmation, emit JSON
uv run nl-db query "list 3 most recent transactions" \
    --db tests/fixtures/sample.db --no-confirm --json

# Disable auto-LIMIT
uv run nl-db query "list every transaction" \
    --db tests/fixtures/sample.db --limit 0
```

Every run is logged as JSONL to `~/.local/share/nl-db/logs/YYYYMMDD.jsonl`.

Writes are refused by default. Enable with `--allow-writes` (still requires confirmation):

```bash
uv run nl-db query "delete transaction 99" --db ... --allow-writes
```

## 4. Use as an MCP server

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nl-db": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/nl-db",
        "run", "nl-db-mcp",
        "--db", "/absolute/path/to/your.db"
      ],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Restart Claude Desktop. You should see four tools — `list_tables`, `describe_database`, `describe_schema`, `query_database` — and two resources (`db://schema` for the full schema, `db://schema/<table>` per-table) in the conversation.

The NL-only surface is intentional: Claude shouldn't write SQL when nl-db can do that for it. To expose the raw-SQL escape hatch (`run_sql`), add `"--expose-run-sql"` to the `args` list. To allow writes through `run_sql`, add **both** `"--expose-run-sql"` and `"--allow-writes"` (the latter requires the former — writes are reachable only through `run_sql`). Tool annotations will then advertise `destructiveHint: true` so Claude prompts before execution.

### Cursor

Cursor reads MCP servers from `~/.cursor/mcp.json`. Same shape as above.

### Verifying

Ask the host model: *"What tables are in the database?"* — it should call `list_tables` and report back. Then: *"How much did Alice spend last month?"* — it should call `query_database`.

## 5. Eval harness

Run the 30-question evaluation against the configured provider:

```bash
# All 30 questions
uv run python -m eval.runner

# Quick smoke test
uv run python -m eval.runner --limit 5

# Override provider for this run only
uv run python -m eval.runner --provider openai_compatible
```

Reports go to `eval/reports/YYYYMMDD_HHMMSS.md` — a Markdown table with the generated SQL and pass/fail reason for every case.

This harness is load-bearing for the project: every prompt change or provider swap should be re-evaluated against it.
