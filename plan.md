# NL-to-SQL + MCP Server — Project Plan

> Natural language database querying powered by Apple Intelligence, with MCP server support.

---

## Vision

Build a privacy-first, cost-efficient tool that lets any user query databases in plain English — running entirely on-device via Apple Intelligence, and exposable as an MCP server to any compatible client (Claude Desktop, Cursor, etc.).

---

## Architecture

```
[Natural Language Input]
        ↓
[LLM Layer] ← Apple Intelligence via OpenAI-compatible Swift HTTP server
        ↓          (fallback: any OpenAI-compatible API)
[SQL Generator] ← schema-aware prompt builder
        ↓
[SQL Validator] ← syntax check, DML guard, timeout
        ↓
[Query Executor] ← read-only by default
        ↓
[Result Formatter] ← table / JSON / NL summary
        ↓
[MCP Server Layer] ← exposes tools to external clients
```

### Key Design Decisions

- **Schema-first prompting** — always fetch live schema before generating SQL; inject it into every prompt
- **SQL transparency** — show generated SQL to user before execution; never auto-run silently
- **Abstracted LLM layer** — OpenAI-compatible interface means any model can be swapped in via config
- **Read-only by default** — writes require an explicit `--allow-writes` flag

---

## Components

### 1. Apple Intelligence HTTP Server (Swift — separate project)
- Wraps on-device model, exposes `POST /v1/chat/completions`
- Request/response matches OpenAI spec exactly
- This is a dependency, not built here — must be running locally

### 2. Schema Extractor
- Connects to DB, extracts tables, columns, types, PKs, FKs
- Outputs token-efficient schema string for prompt injection
- Supported DBs: SQLite (phase 1), PostgreSQL, MySQL (phase 4)

### 3. SQL Generator
- Builds prompt: system message + schema + few-shot examples + user question
- Calls LLM via HTTP, parses SQL from response (handles markdown fences)
- Returns SQL string for validation — does not execute directly

### 4. SQL Validator
- Syntax check before execution
- Reject or flag DML (INSERT/UPDATE/DELETE) unless writes are allowed
- Enforce query timeout (default: 10s)

### 5. Query Executor
- Executes validated SQL against connected DB
- Result formats: table (CLI), JSON (API/MCP), NL summary (second LLM pass)
- Connection pooling for repeated queries

### 6. MCP Server
Exposes the pipeline as an MCP server with four tools:

| Tool | Description |
|------|-------------|
| `list_tables` | Returns all table names in the connected DB |
| `describe_schema(table_name)` | Returns full schema for a specific table |
| `query_database(question)` | Full NL → SQL → result pipeline |
| `run_sql(sql)` | Executes raw SQL directly (gated behind `--allow-writes`) |

> Tool descriptions must be written precisely — the host model uses them to decide when and how to call each tool. Treat them as product copy, not boilerplate.

---

## Use Cases

1. **Personal finance** — query a local SQLite transactions DB ("how much did I spend on groceries last month?")
2. **Developer DB exploration** — explore an unfamiliar schema from inside Cursor or Claude Desktop without leaving the editor
3. **Small business ops** — non-technical operators querying sales/inventory data without SQL knowledge
4. **Privacy-sensitive domains** — healthcare, legal, HR — data never leaves the device, no cloud API required
5. **Local dev & testing** — verify seed data, check FK relationships, audit state after test runs
6. **MCP client augmentation** — any MCP-compatible client gains live DB access with zero additional work

---

## Phased Plan

### Phase 1 — Foundation (Weeks 1–3)
Goal: end-to-end pipeline working for SQLite via CLI

- [ ] Project structure and dependency setup
- [ ] SQLite schema extractor (tables, columns, types, PKs, FKs)
- [ ] Prompt builder (system + schema + few-shot + question)
- [ ] Wire to a cloud LLM first (GPT-4 or Claude) for baseline quality testing
- [ ] SQL validator (syntax, DML guard, timeout enforcement)
- [ ] Query executor with table + JSON output
- [ ] CLI: connect → ask → show SQL → confirm → run → display results
- [ ] Integration tests against a sample SQLite DB

### Phase 2 — Apple Intelligence Integration (Weeks 4–5)
Goal: replace cloud LLM with on-device model; validate quality

- [ ] Define LLM provider interface (abstraction layer)
- [ ] Implement OpenAI-compatible HTTP client
- [ ] Point client at Apple Intelligence Swift server
- [ ] Run Phase 1 test suite against on-device model; document quality gaps
- [ ] Tune prompts specifically for Apple Intelligence's capabilities and context window
- [ ] Implement fallback: if SQL fails validation, retry with cloud model
- [ ] Benchmark latency (cold start, warm, complex queries)

### Phase 3 — MCP Server (Weeks 6–8)
Goal: expose pipeline as a working MCP server; verify with real clients

- [ ] MCP server scaffold using official MCP SDK
- [ ] Implement `list_tables` tool
- [ ] Implement `describe_schema` tool
- [ ] Implement `query_database` tool (full pipeline)
- [ ] Implement `run_sql` tool (behind `--allow-writes` flag)
- [ ] Write tool descriptions (first-class work, not an afterthought)
- [ ] Add `dry_run` mode: generate SQL but don't execute, return SQL for review
- [ ] Integration test with Claude Desktop
- [ ] Integration test with Cursor

### Phase 4 — Polish & Expand (Weeks 9–12)
Goal: multi-DB support, NL result summaries, production readiness

- [ ] PostgreSQL adapter
- [ ] MySQL/MariaDB adapter
- [ ] NL result summarizer (second LLM pass over query results)
- [ ] Connection config manager (multiple named DB connections)
- [ ] Schema caching + prompt caching for performance
- [ ] Optional: simple web UI (query input, SQL preview, results table)
- [ ] User-facing docs and setup guide

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Apple Intelligence SQL quality insufficient for complex queries | High | Prompt engineering + few-shot examples; fallback to cloud model on validation failure |
| Wrong SQL silently returns incorrect results | High | SQL transparency step is mandatory; add row count warnings and empty result alerts |
| Schema too large for on-device context window | Medium | Schema compression; omit indexes; let user specify relevant tables |
| Vague MCP tool descriptions cause host model misuse | Medium | Treat descriptions as product copy; test with multiple host models; add `dry_run` mode |
| SQL injection via generated queries | Medium | Read-only by default; validate and sanitize all SQL before execution |
| Apple Intelligence Swift server reliability | Low | Abstract LLM behind interface; cloud fallback always available |

---

## Success Metrics

**Phase 1:** ≥80% SQL accuracy on 30-query test set; P90 latency <5s; zero silent wrong results  
**Phase 2:** On-device accuracy within 15pp of cloud baseline; fallback triggers correctly  
**Phase 3:** All four tools callable from Claude Desktop; no data leaves device on Apple Intelligence path  
**Phase 4:** Postgres + MySQL tests passing; NL summaries rated useful; setup completable in <30 min  

---

## Tech Stack

- **Language:** Swift (natural fit for Apple Intelligence integration) or Python for SQL engine
- **LLM interface:** OpenAI-compatible HTTP (abstracted — swap any compliant endpoint via config)
- **Primary model:** Apple Intelligence via companion Swift HTTP server
- **Fallback model:** Any OpenAI-compatible API (GPT-4, Claude, Ollama)
- **MCP SDK:** Official Anthropic MCP SDK
- **DB drivers:** SQLite (built-in), libpq (PostgreSQL), MySQL connector
- **Testing:** unit tests per component + integration tests against in-memory SQLite
