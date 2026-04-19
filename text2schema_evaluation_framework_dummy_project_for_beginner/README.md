# SchemaEval

A JSON Output Quality Evaluation Framework that compares LLM-generated JSON against golden reference JSON using **7 complementary metrics** and visualises results in an interactive dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the Project](#running-the-project)
- [Optional: Ollama LLM Judge](#optional-ollama-llm-judge)
- [CLI Usage](#cli-usage)
- [Running Tests](#running-tests)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

SchemaEval evaluates the quality of LLM-generated JSON by scoring it against a golden reference across 7 metrics:

| Metric | Description |
|---|---|
| **Jaccard** | Set overlap of flattened `key=value` pairs |
| **Cosine** | TF-IDF cosine similarity on JSON token streams |
| **Levenshtein** | Normalised edit distance on serialised JSON |
| **BLEU** | Sentence BLEU on JSON token sequences |
| **ROUGE-L** | ROUGE-L F-measure on serialised JSON |
| **Field Diff** | Recursive field-level comparison (matched / missing / extra / mismatch) |
| **LLM Judge** | Semantic scoring via Ollama (optional — system works without it) |

A **composite score** (weighted average of all 7) determines pass/fail against a configurable threshold.

---

## Architecture

```
React Frontend (Vite, port 5173)
        │  /api proxy
        ▼
FastAPI Backend (port 8000)
        │
        ├── EvaluationEngine   ← async orchestration, LRU cache
        │       ├── 7 Scorers  ← CPU-bound run in ThreadPoolExecutor
        │       └── LLM Judge  ← async HTTP to Ollama
        └── SQLite DB          ← stores all run results (aiosqlite)
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | `python3 --version` |
| pip | ≥ 23 | `pip --version` |
| Node.js | ≥ 18 | `node --version` |
| npm | ≥ 9 | `npm --version` |
| Ollama | any | Optional — only needed for LLM Judge metric |

---

## Project Structure

```
schemaeval/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── schemaeval/                  ← Python package
│   ├── config.py
│   ├── models/                  ← Pydantic data models
│   ├── scorers/                 ← 7 metric scorer classes
│   ├── engine/                  ← LRU cache + async evaluator
│   ├── db/                      ← SQLAlchemy async SQLite
│   ├── api/                     ← FastAPI app + routes + WebSocket
│   └── cli/                     ← argparse CLI
├── golden_data/
│   └── samples/                 ← 3 reference JSON files
├── frontend/                    ← React 18 + TypeScript + Vite
│   └── src/
│       ├── types/               ← TypeScript mirrors of Pydantic models
│       ├── api/                 ← axios client
│       ├── hooks/               ← useWebSocket hook
│       ├── components/          ← KpiCards, RadarChart, TrendChart, FieldDiffViewer, EvalForm
│       └── pages/               ← Dashboard, History
└── tests/
    ├── unit/scorers/            ← 1 file per scorer, ≥5 cases each
    ├── integration/             ← full API pipeline tests
    └── property/                ← Hypothesis: reflexivity, symmetry, boundedness
```

---

## Backend Setup

### 1. Clone / enter the project directory

```bash
cd text2schema_evaluation_framework_dummy_project_for_beginner
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development (includes pytest, hypothesis, etc.):

```bash
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` if you need to change defaults (the defaults work out of the box):

```env
DB_URL=sqlite+aiosqlite:///./schemaeval.db
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
HOST=0.0.0.0
PORT=8000
```

---

## Frontend Setup

### 1. Install Node dependencies

```bash
cd frontend
npm install
cd ..
```

> **Note:** If you are on Node < v20.19, use `npm create vite@5` if re-scaffolding is ever needed.

---

## Running the Project

### Start the backend API server

```bash
# From the project root
python -m schemaeval serve --reload --port 8000
```

The API is now available at:
- **REST API:** `http://localhost:8000`
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Start the frontend dev server

Open a **second terminal**:

```bash
cd frontend
npm run dev
```

The dashboard is now available at **`http://localhost:5173`**.

The Vite dev server proxies all `/api` and `/ws` requests to `localhost:8000` — no CORS issues.

### Verify everything is working

```bash
# Health check
curl http://localhost:8000/api/health

# Quick evaluation (identical JSON → should score ~0.86, passed: true)
curl -s -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"generated": {"name": "Alice", "age": 30}, "golden": {"name": "Alice", "age": 30}}' \
  | python3 -m json.tool
```

---

## Optional: Ollama LLM Judge

The system works fully without Ollama (the LLM Judge metric returns `0.0` gracefully when unavailable). To enable it:

```bash
# 1. Install Ollama — https://ollama.com
# 2. Start the Ollama daemon
ollama serve

# 3. Pull the model (one-time download, ~4 GB)
ollama pull llama3

# 4. Verify
curl http://localhost:11434/api/tags
```

Once running, the `/api/health` endpoint will show `"ollama": {"available": true}`.

To exclude the LLM Judge from scoring, set its weight to `0` and redistribute to other metrics:

```json
{
  "generated": { ... },
  "golden": { ... },
  "metric_config": {
    "jaccard": 0.2,
    "cosine": 0.2,
    "levenshtein": 0.2,
    "bleu": 0.15,
    "rouge": 0.15,
    "field_diff": 0.1,
    "llm_judge": 0.0
  }
}
```

> Weights must sum to exactly `1.0`.

---

## CLI Usage

### Evaluate two JSON files

```bash
# Create sample files
echo '{"name": "Alice", "role": "admin"}' > /tmp/generated.json
echo '{"name": "Alice", "role": "user"}' > /tmp/golden.json

# Run evaluation — prints JSON result to stdout
python -m schemaeval eval \
  --generated /tmp/generated.json \
  --golden /tmp/golden.json \
  --threshold 0.7
```

### Start the server via CLI

```bash
python -m schemaeval serve --host 0.0.0.0 --port 8000 --reload
```

---

## Running Tests

### Run the full test suite

```bash
# From the project root
pytest tests/ -v
```

### Run specific test categories

```bash
# Unit tests only (44 tests — one file per scorer)
pytest tests/unit/ -v

# Integration tests only (8 tests — full API pipeline with in-memory SQLite)
pytest tests/integration/ -v

# Property-based tests only (9 tests — Hypothesis)
pytest tests/property/ -v
```

### Run with coverage

```bash
pytest tests/ --cov=schemaeval --cov-report=term-missing
```

### Expected output

```
61 passed, 1 warning in ~4s
```

The single warning (`python_multipart`) is from the installed version of Starlette and does not affect functionality.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Server + Ollama availability status |
| `POST` | `/api/evaluate` | Evaluate a single generated/golden JSON pair |
| `POST` | `/api/evaluate/batch` | Evaluate a list of pairs in one request |
| `GET` | `/api/results` | List recent evaluation runs (default last 50) |
| `GET` | `/api/results/{run_id}` | Get a specific run by ID |
| `WS` | `/ws/evaluate` | Real-time evaluation via WebSocket |

#### POST `/api/evaluate` — request body

```json
{
  "generated": { "name": "Alice", "age": 30 },
  "golden":    { "name": "Alice", "age": 30 },
  "pass_threshold": 0.7,
  "run_id": "optional-custom-id",
  "tags": ["prod", "v2"],
  "metric_config": {
    "jaccard": 0.142857,
    "cosine": 0.142857,
    "levenshtein": 0.142857,
    "bleu": 0.142857,
    "rouge": 0.142857,
    "field_diff": 0.142857,
    "llm_judge": 0.142858
  }
}
```

Full interactive docs available at `http://localhost:8000/docs`.

---

## Configuration

All settings are read from environment variables or a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `DB_URL` | `sqlite+aiosqlite:///./schemaeval.db` | Database connection string |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `llama3` | Model name to use for LLM Judge |
| `HOST` | `0.0.0.0` | API server bind address |
| `PORT` | `8000` | API server port |
| `CACHE_MAX_SIZE` | `256` | In-memory LRU cache capacity |
| `DEFAULT_PASS_THRESHOLD` | `0.7` | Default composite score pass threshold |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'schemaeval'`**
Run all commands from the project root directory, not from inside `schemaeval/`.

**`pytest: no tests found`**
Same as above — run `pytest tests/` from the project root.

**Port 8000 already in use**
```bash
python -m schemaeval serve --port 8001
# Update frontend/vite.config.ts proxy target to match
```

**Ollama not available / LLM Judge score is 0.0**
This is expected behaviour when Ollama is not running. All other 6 metrics still work normally. See [Optional: Ollama LLM Judge](#optional-ollama-llm-judge) to enable it.

**Frontend `npm run dev` shows blank page**
Ensure the backend is running on port 8000 before loading the frontend. The dashboard makes an API call on first render.

**Node version error when installing `create-vite`**
Use `npm create vite@5` — the latest `create-vite` requires Node ≥ v20.19.
