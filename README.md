# nl-db

Natural language database querying. Privacy-first by design (on-device via Apple Intelligence), provider-agnostic by construction (any OpenAI-compatible endpoint, plus Anthropic and OpenAI directly), and exposable as an MCP server to any compatible client.

## Quickstart

```bash
uv sync

# Configure your provider (any one of these is enough)
echo "ANTHROPIC_API_KEY=sk-..." >> .env       # for Claude
# OR
echo "OPENAI_API_KEY=sk-..." >> .env          # for GPT
# OR (for Ollama / Apple Intelligence shim / vLLM)
export NL_DB_PROVIDER__NAME=openai_compatible
export NL_DB_PROVIDER__BASE_URL=http://localhost:8080/v1
export NL_DB_PROVIDER__MODEL=local-llama

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
3. Validates with `sqlglot` and auto-injects `LIMIT` for unbounded reads
4. Shows you the SQL **plus** a one-sentence plain-English paraphrase
5. Waits for your confirmation (skip with `--no-confirm`)
6. Executes and prints a table (or JSON with `--json`)
7. Logs the run as JSONL to `~/.local/share/nl-db/logs/`

Writes (`INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`TRUNCATE`) are refused unless you pass `--allow-writes`.

## Configuration

Precedence: env vars > `.env` > `~/.config/nl-db/config.toml` > defaults.

A full `config.toml` looks like:

```toml
[provider]
name = "anthropic"            # "anthropic" | "openai" | "openai_compatible"
model = "claude-sonnet-4-6"
# base_url = "http://localhost:8080/v1"   # required if name = "openai_compatible"

[limits]
max_rows = 1000
timeout_s = 10.0
max_prompt_tokens = 8000
```

API keys come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENAI_COMPATIBLE_API_KEY` — never write them into `config.toml`.

## Project state

This is an active build. See `plan.md` for the full design.

- [x] Config layer (env / .env / TOML precedence)
- [x] Schema extractor + cache (SQLite)
- [x] LLM provider abstraction (Anthropic, OpenAI, OpenAI-compatible)
- [x] Prompt builder + few-shot + paraphrase
- [x] SQL generator + sqlglot validator + auto-LIMIT
- [x] Query executor + table/JSON formatters
- [x] Pipeline orchestrator
- [x] CLI (`nl-db query`, `nl-db schema`, `nl-db config`)
- [ ] Sample database fixture
- [ ] 30-question eval harness
- [ ] MCP stdio server
- [ ] Setup docs (Apple Intelligence shim, Claude Desktop integration)
