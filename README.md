# nl-db

Natural language database querying. Privacy-first by design (on-device via Apple Intelligence), provider-agnostic by construction (any OpenAI-compatible endpoint, plus Anthropic and OpenAI directly), and exposable as an MCP server to any compatible client.

> **Status:** under active construction. See `plan.md` for the design and the build plan.

## Quickstart

```bash
uv sync
cp .env.example .env  # fill in your API key
uv run nl-db query "list the 5 most recent transactions" --db tests/fixtures/sample.db
```

Full docs in `docs/setup.md` once the CLI ships.
