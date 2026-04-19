# Text2Schema

Convert a plain-English description of your data domain into a production-ready database schema — complete with DDL SQL, ERD diagram, and a quality score — powered by a **local Ollama LLM** (`llama3.2:3b`). No cloud API keys, no data leaves your machine.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [Architecture Deep-Dive](#architecture-deep-dive)
5. [Quick Start (Docker)](#quick-start-docker)
6. [Local Development Setup](#local-development-setup)
7. [Running Tests](#running-tests)
8. [API Reference](#api-reference)
9. [Configuration Reference](#configuration-reference)
10. [Environment Variables](#environment-variables)
11. [Troubleshooting](#troubleshooting)

---

## What It Does

You send a sentence like:

> "E-commerce platform with users, products, orders, and reviews"

and get back:

- **DDL SQL** (PostgreSQL / MySQL) or **JSON Schema** (MongoDB / DynamoDB)
- A **quality score** across 4 dimensions (syntax, integrity, naming, completeness)
- An **ERD diagram** (React Flow nodes + edges) rendered in the browser
- **Automatic retry** — if the quality score is below 0.8, the pipeline re-prompts the LLM with targeted feedback, up to 3 times

---

## System Requirements

| Tool | Minimum Version | Notes |
|---|---|---|
| Python | 3.11+ | Backend |
| Node.js | 20+ | Frontend build |
| Docker + Docker Compose | v2+ | Full stack |
| Ollama | latest | Runs on host, not in Docker |
| RAM | 4 GB+ | llama3.2:3b needs ~2 GB |
| Storage | 4 GB+ | Model weights |

**Model options** (change `primary_model` / `judge_model` in `config.yaml` to switch):

| Model | Disk | Min RAM | Best for |
|---|---|---|---|
| `llama3.2:3b` | ~2 GB | 4 GB | **Default — runs on any laptop** |
| `llama3.1:8b` | ~5 GB | 8 GB | Better quality, most dev machines |
| `llama3.3:70b` | ~40 GB | 48 GB | Production quality, GPU workstation |

---

## Project Structure

```
text2schema/
│
├── config.yaml                  # All tunable parameters (LLM, quality, cache, DB, server)
├── .env.example                 # Template for environment variables
├── requirements.txt             # Python dependencies
├── Dockerfile                   # API container
├── docker-compose.yml           # 4-service stack (api, frontend, redis, postgres-sandbox)
├── Makefile                     # Developer shortcuts
├── pytest.ini                   # Test runner config (asyncio_mode = auto)
│
├── app/
│   ├── config.py                # Pydantic Settings — loads config.yaml + env var overrides
│   ├── main.py                  # FastAPI app, CORS, startup event
│   ├── models/
│   │   └── schema.py            # All Pydantic data models (single source of truth)
│   ├── api/
│   │   ├── routes.py            # POST /generate, GET /health, /providers, /dialects
│   │   └── dependencies.py      # lru_cache singleton factories for all core classes
│   └── core/
│       ├── cache_manager.py     # Redis (or in-memory) cache — context + response caching
│       ├── prompt_engine.py     # Jinja2 template renderer + knowledge YAML loader
│       ├── llm_router.py        # Ollama ChatOllama wrapper with exponential backoff retry
│       ├── output_parser.py     # Strips markdown fences, validates JSON → SchemaDefinition
│       ├── schema_converter.py  # SchemaDefinition → DDL SQL / MongoDB JSON / DynamoDB JSON
│       ├── erd_generator.py     # SchemaDefinition → React Flow nodes + edges
│       ├── quality_evaluator.py # 4-dimension scorer: syntax, integrity, naming, completeness
│       └── retry_handler.py     # Orchestrates the generate → evaluate → retry loop
│
├── knowledge/
│   ├── postgresql/conventions.yaml
│   ├── mysql/conventions.yaml
│   ├── mongodb/conventions.yaml
│   └── dynamodb/conventions.yaml   # DB-specific naming rules, type maps, few-shot examples
│
├── templates/
│   ├── generate.j2              # Prompt template for schema generation
│   └── judge.j2                 # Prompt template for completeness scoring
│
├── tests/
│   ├── conftest.py              # Shared fixtures (sample_schema, mock_llm, mock_cache, etc.)
│   ├── unit/                    # Pure unit tests — no external services needed
│   ├── integration/             # API-level tests — mock LLM via dependency_overrides
│   └── property/                # Hypothesis-based property tests
│
└── frontend/
    ├── src/
    │   ├── App.tsx              # Root component — state, layout, view toggle
    │   ├── api/client.ts        # Axios instance + TypeScript interfaces
    │   └── components/
    │       ├── SchemaInput.tsx  # Description textarea, DB selector, format selector
    │       ├── QualityBadge.tsx # Score bars (green ≥0.8, yellow ≥0.6, red <0.6)
    │       ├── OutputPanel.tsx  # CodeMirror editor with SQL syntax highlighting
    │       └── ERDViewer.tsx    # React Flow canvas with custom TableNode
    └── vite.config.ts           # Dev proxy: /api → http://localhost:8000
```

---

## Architecture Deep-Dive

### 7-Layer Pipeline

```
User Input (POST /api/generate)
    │
    ▼
1. PromptEngine
   - Loads knowledge YAML for the target DB (postgresql/mysql/mongodb/dynamodb)
   - Caches rendered knowledge context in Redis (TTL 1 h) using SHA-256 of file content
   - Renders generate.j2 Jinja2 template with description + context + error_context (on retry)

    │
    ▼
2. LLMRouter
   - Sends prompt to ChatOllama (llama3.2:3b) at http://localhost:11434
   - Exponential backoff: delay = backoff_base × 2^attempt + jitter (up to max_retries = 3)
   - No cloud fallbacks — raises RuntimeError if all retries fail

    │
    ▼
3. OutputParser
   - Strips ```json ... ``` fences from raw LLM response
   - Extracts outermost { ... } block
   - Validates against SchemaDefinition Pydantic model
   - On ValidationError: formats field errors as human-readable strings fed back to retry loop

    │
    ▼
4. QualityEvaluator  ◄──────────────────────────────────────┐
   Runs 4 dimensions in parallel:                           │
                                                            │
   a) Syntax (weight 0.25)                                  │
      - PostgreSQL: spins up a sandbox schema in asyncpg,   │
        executes DDL, drops it in finally block             │
      - Returns 0.5 (not 0.0) if Postgres is unreachable    │
      - Non-SQL targets: always 1.0                         │
                                                            │
   b) Integrity (weight 0.25)                               │
      - Every table has a PK                                │
      - All FK references point to real tables + columns    │
      - No duplicate column names per table                 │
      - Deducts 0.1 per issue, floor 0.0                    │
                                                            │
   c) Naming (weight 0.15)                                  │
      - snake_case regex: ^[a-z][a-z0-9_]*$                │
      - Not a SQL reserved word (60+ words checked)         │
      - Length ≤ 63 chars                                    │
      - Deducts 0.05 per issue, floor 0.0                   │
                                                            │
   d) Completeness (weight 0.35)                            │
      - Calls LLM judge (same Ollama model, temp=0.0)       │
        with judge.j2 template                              │
      - Judge scores: entity_coverage×0.40                  │
                    + relationship_accuracy×0.35            │
                    + attribute_completeness×0.25           │
      - Returns 0.5 on parse failure (partial credit)       │
                                                            │
   Composite = syntax×0.25 + integrity×0.25                 │
             + naming×0.15 + completeness×0.35              │
                                                            │
   If composite < 0.8 AND attempt < max_retry (3):          │
      Build error_context from failing dimensions ──────────┘
      (re-enters loop at step 1 with error_context injected)

    │  (composite ≥ 0.8 OR retries exhausted)
    ▼
5. SchemaConverter
   - postgresql → CREATE TABLE DDL with FK CONSTRAINT + CREATE INDEX
   - mysql      → DDL with ENGINE=InnoDB, utf8mb4, BIGINT UNSIGNED AUTO_INCREMENT
   - mongodb    → {collection: {$jsonSchema: {...}}} JSON
   - dynamodb   → [{TableName, AttributeDefinitions, KeySchema, BillingMode}] JSON

    │
    ▼
6. ERDGenerator
   - Builds React Flow nodes (type="tableNode", grid layout 3 cols × 320px × 280px)
   - Builds edges from FK columns (sourceHandle/targetHandle = column name — must match
     <Handle id={col.name}> in ERDViewer.tsx TableNode component)

    │
    ▼
7. Response cached in Redis (TTL 24 h), returned as GenerateResponse JSON
```

### Quality Score Formula

```
composite = (syntax × 0.25) + (integrity × 0.25) + (naming × 0.15) + (completeness × 0.35)
passed    = composite >= 0.8
```

Weights are hardcoded in `app/models/schema.py:QualityScore` — **do not import `settings` there** (circular import).

### Caching Strategy

| Cache Key Prefix | Content | TTL |
|---|---|---|
| `ctx:` | Rendered knowledge context for a DB dialect | 1 hour |
| `resp:` | Full GenerateResponse JSON for a (description, db, format) tuple | 24 hours |

---

## Quick Start (Docker)

This is the fastest path. Ollama still runs on your **host machine**, not inside Docker.

### Step 1 — Install and start Ollama

```bash
# macOS
brew install ollama

# Linux — see https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
# Pull the model (one-time, ~40 GB)
ollama pull llama3.2:3b

# Keep Ollama running in a separate terminal
ollama serve
```

Verify:
```bash
curl http://localhost:11434/api/tags
# Should include "llama3.2:3b" in the response
```

### Step 2 — Clone and configure

```bash
git clone <repo-url>
cd text2schema
cp .env.example .env
# Default values in .env work out of the box — no edits needed
```

### Step 3 — Start all services

```bash
make up
# Starts: api (8000), frontend (5173), redis (6379), postgres-sandbox (5433)
```

Watch logs:
```bash
make logs
```

### Step 4 — Verify everything is running

```bash
# API health
curl http://localhost:8000/api/health
# {"status":"ok"}

# Active LLM provider
curl http://localhost:8000/api/providers
# [{"name":"ollama","model":"llama3.2:3b","role":"primary"}]

# Open the UI
open http://localhost:5173
```

### Step 5 — Generate a schema

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "E-commerce platform with users, products, orders, and reviews",
    "target_db": "postgresql",
    "output_format": "sql"
  }'
```

Expected response shape:
```json
{
  "schema": { "tables": [...], "target_db": "postgresql", "version": "1.0" },
  "quality": {
    "syntax": 1.0,
    "integrity": 1.0,
    "naming": 1.0,
    "completeness": 0.87,
    "composite": 0.93,
    "passed": true
  },
  "retry_count": 0,
  "outputs": { "sql": "CREATE TABLE ...", "erd": "{...}" },
  "cached": false,
  "processing_time_ms": 12400.5
}
```

### Step 6 — Tear down

```bash
make down        # stop containers, keep volumes
make clean       # stop containers + delete volumes + clear pycache
```

---

## Local Development Setup

Use this path when you want to edit Python code with hot-reload, without rebuilding Docker images.

### Step 1 — Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

make install
# equivalent to: pip install -r requirements.txt
```

### Step 2 — Start backing services (Redis + Postgres only)

```bash
docker-compose up -d redis postgres-sandbox
```

### Step 3 — Configure environment

```bash
cp .env.example .env
# Default values work — OLLAMA_BASE_URL=http://localhost:11434 is correct for local dev
```

### Step 4 — Start the API with hot-reload

```bash
make dev
# equivalent to: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5 — Start the frontend (optional)

```bash
make frontend-install   # npm install inside ./frontend
make frontend-dev       # Vite dev server at http://localhost:5173
```

Vite proxies `/api/*` → `http://localhost:8000` in dev mode (see `frontend/vite.config.ts`).

---

## Running Tests

### Unit + Property tests (no services needed)

These run entirely in-process — no Ollama, no Redis, no Postgres required.

```bash
make test-unit
# runs: pytest tests/unit tests/property -v
# expect: 33 passed
```

What is tested:

| File | What it covers |
|---|---|
| `tests/unit/test_output_parser.py` | Clean JSON, fenced JSON, invalid JSON, schema shape errors |
| `tests/unit/test_quality_evaluator.py` | Missing PK, bad FK reference, reserved word names, camelCase rejection |
| `tests/unit/test_schema_converter.py` | PostgreSQL DDL, FK constraints, MongoDB JSON Schema, DynamoDB JSON |
| `tests/unit/test_erd_generator.py` | Node count = table count, FK edges created, node structure |
| `tests/unit/test_retry_handler.py` | Success on first attempt (retry_count=0), retry on quality failure |
| `tests/property/test_schema_properties.py` | Hypothesis: DDL always has `CREATE TABLE {name}`, ERD node count invariant, parser never raises |

### Integration tests (requires running API + mocked LLM)

```bash
make test-integration
# runs: pytest tests/integration -v
```

These use `app.dependency_overrides` to replace `get_llm_router` with an `AsyncMock` — no real Ollama call is made.

### Full test suite

```bash
make test
# runs: pytest tests/ -v
```

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /api/health`

```bash
curl http://localhost:8000/api/health
```
```json
{"status": "ok"}
```

### `GET /api/providers`

Returns the active LLM provider.

```bash
curl http://localhost:8000/api/providers
```
```json
[{"name": "ollama", "model": "llama3.2:3b", "role": "primary"}]
```

### `GET /api/dialects`

Returns supported databases.

```bash
curl http://localhost:8000/api/dialects
```
```json
[
  {"id": "postgresql", "label": "PostgreSQL", "output": "sql"},
  {"id": "mysql",      "label": "MySQL",      "output": "sql"},
  {"id": "mongodb",    "label": "MongoDB",    "output": "nosql"},
  {"id": "dynamodb",   "label": "DynamoDB",   "output": "nosql"}
]
```

### `POST /api/generate`

**Request body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `description` | string | yes | — | Min 10 chars. Plain-English description of the domain |
| `target_db` | string | no | `postgresql` | One of: `postgresql`, `mysql`, `mongodb`, `dynamodb` |
| `output_format` | string | no | `sql` | One of: `sql`, `nosql`, `erd`, `all` |
| `conversation_history` | array | no | `[]` | Previous `[{role, content}]` turns for multi-turn refinement |
| `use_cache` | bool | no | `true` | Return cached response if available |

**Example — PostgreSQL:**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Hospital system with patients, doctors, appointments, prescriptions",
    "target_db": "postgresql",
    "output_format": "sql"
  }'
```

**Example — MongoDB:**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Blog platform with authors, posts, tags, and comments",
    "target_db": "mongodb",
    "output_format": "nosql"
  }'
```

**Example — all formats at once:**
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "SaaS CRM with companies, contacts, deals, and activities",
    "target_db": "postgresql",
    "output_format": "all"
  }'
```

**Response fields:**

| Field | Description |
|---|---|
| `schema` | Structured `SchemaDefinition` (tables, columns, FK definitions) |
| `quality.syntax` | DDL executed cleanly in sandbox (0.0–1.0) |
| `quality.integrity` | All PKs present, FKs valid, no duplicate columns (0.0–1.0) |
| `quality.naming` | snake_case, no reserved words, length ≤ 63 (0.0–1.0) |
| `quality.completeness` | LLM judge evaluates entity/relationship/attribute coverage (0.0–1.0) |
| `quality.composite` | Weighted composite score (threshold 0.8) |
| `quality.passed` | `true` if composite ≥ 0.8 |
| `retry_count` | How many times the pipeline retried (0–3) |
| `outputs` | Dict of format → string (`"sql"`, `"nosql"`, `"erd"`) |
| `cached` | Whether this response came from Redis cache |
| `processing_time_ms` | Wall-clock time in milliseconds |

---

## Configuration Reference

All settings live in `config.yaml`. Environment variables override YAML values.

```yaml
llm:
  primary: ollama
  primary_model: llama3.2:3b       # Model used for schema generation
  judge_model: llama3.2:3b         # Model used for completeness judging
  ollama_base_url: "http://localhost:11434"
  fallback_order: []                # No fallbacks — Ollama only
  temperature: 0.2                  # Low temp for deterministic output
  max_tokens: 4096
  max_retries: 3                    # Ollama connection retries (not quality retries)
  backoff_base: 1.0                 # delay = backoff_base × 2^attempt + jitter

quality:
  threshold: 0.8                    # Minimum composite score to accept output
  weights:
    syntax: 0.25
    integrity: 0.25
    naming: 0.15
    completeness: 0.35
  max_retry: 3                      # Max quality-retry loops

cache:
  backend: redis
  context_ttl: 3600                 # Knowledge context cache: 1 hour
  response_ttl: 86400               # Full response cache: 24 hours
  redis_url: "redis://redis:6379/0"

database:
  sandbox_dsn: "postgresql://sandbox:sandbox@postgres-sandbox:5432/sandbox"
  sandbox_schema_prefix: "sandbox_" # Temporary schemas: sandbox_<uuid8>

server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["http://localhost:5173"]
  log_level: info
```

---

## Environment Variables

Copy `.env.example` to `.env`. All values have working defaults.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Override Ollama host (e.g. remote GPU machine) |
| `REDIS_URL` | `redis://localhost:6379/0` | Override Redis connection string |
| `SANDBOX_DB_DSN` | `postgresql://sandbox:sandbox@localhost:5433/sandbox` | Override Postgres sandbox DSN |

When running via `make up` (Docker), the `api` container uses:
- `OLLAMA_BASE_URL=http://host.docker.internal:11434` (reaches Ollama on the host)
- `extra_hosts: host.docker.internal:host-gateway` (Linux compatibility)

---

## Troubleshooting

**Ollama connection refused**
```
RuntimeError: Ollama generate failed after 3 retries
```
→ Make sure `ollama serve` is running in a separate terminal.

**Model not found**
```
Error: model "llama3.2:3b" not found
```
→ Run `ollama pull llama3.2:3b` (requires ~40 GB free disk space).

**Docker container cannot reach Ollama**
```
Connection refused to host.docker.internal:11434
```
→ On Linux, ensure Docker Compose includes `extra_hosts: ["host.docker.internal:host-gateway"]` (already set). On Mac, Docker Desktop resolves `host.docker.internal` natively.

**Postgres sandbox unreachable (syntax score = 0.5)**
```
"Sandbox unreachable: ..."
```
→ The quality evaluator returns `0.5` (not `0.0`) when Postgres is down, so generation still works. Start `postgres-sandbox` with `docker-compose up -d postgres-sandbox`.

**Slow first response**
→ `llama3.2:3b` is a 70-billion parameter model. First-token latency on CPU can be 30–60 s. A GPU with ≥ 48 GB VRAM (e.g. 2× RTX 3090) reduces this to ~2–5 s.

**Unit tests fail with `ModuleNotFoundError: langchain_ollama`**
→ Run `pip install langchain-ollama` or `make install`. Unit tests mock the LLM so Ollama does not need to be running, but the package must be importable.

**Pydantic validation error in config**
→ Check that `config.yaml` has not been hand-edited with incorrect types. The YAML `fallback_order` must be a list (even if empty: `[]`).

**Cache returning stale results**
→ Pass `"use_cache": false` in the request body, or flush Redis:
```bash
docker-compose exec redis redis-cli FLUSHALL
```

---

## Makefile Targets

| Target | What it does |
|---|---|
| `make up` | Start all 4 Docker services in background |
| `make down` | Stop containers (volumes preserved) |
| `make build` | Rebuild Docker images |
| `make logs` | Tail all container logs |
| `make install` | `pip install -r requirements.txt` |
| `make dev` | Run API with hot-reload (uvicorn) |
| `make test` | Full test suite |
| `make test-unit` | Unit + property tests only |
| `make test-integration` | Integration tests only |
| `make frontend-install` | `npm install` inside `./frontend` |
| `make frontend-dev` | Vite dev server at port 5173 |
| `make clean` | Stop + remove volumes + clear pycache |
