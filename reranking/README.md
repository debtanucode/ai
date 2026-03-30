# Semantic Search + Reranking — Hands-on Learning Project

## Project Description

This project is a hands-on, runnable implementation of the concepts covered in the *Reranking* study notes and the *Semantic Search with Language Models* book (Chapter 8). It demonstrates how a real two-stage search pipeline works — from fast first-stage retrieval all the way through precise cross-encoder reranking — and measures how much each stage improves the results using standard information retrieval metrics.

The dataset used throughout is `indian_movies.csv`, a collection of 20 Indian movies with names, descriptions, release dates, and IMDB ratings. The **description** column is used as the search corpus — every concept is grounded in this real data so the output is always meaningful and relatable.

The project is structured so that running a single file (`main.py`) executes all 9 features in sequence — from loading data all the way through MAP evaluation — with clear section headers and printed explanations at every step.

---

## Problem Statement

A search system that only uses keyword matching (BM25) has a critical limitation — it matches **words**, not **meaning**. If a user searches for `"covert spy mission"`, BM25 will miss a movie described as `"deep-cover agent operating in enemy territory"` — zero word overlap means zero score.

This raises four interconnected problems that this project addresses:

| Problem | What It Means | How This Project Addresses It |
|---|---|---|
| BM25 misses semantic matches | Word overlap only — synonyms and paraphrases are invisible | Dense retrieval (bi-encoder) catches meaning-based matches alongside BM25 |
| Dense retrieval misses exact terms | Semantic similarity can blur precise rare words | BM25 runs in parallel to cover exact keyword matches dense retrieval may miss |
| First-stage results are rough | Both BM25 and dense retrieval are fast approximations | Cross-encoder reranker reads query + document together for precise reordering |
| No way to measure improvement | How do you know if reranking actually helped? | Precision@k, Average Precision, and MAP metrics with a 3-query test suite |

---

## How We Are Solving It

The project implements 9 features, run in sequence by `main.py`.

### Feature 1 — Data Loading

The `indian_movies.csv` file is loaded using Python's built-in `csv` module. Two structures are returned:
- `movies` — full list of dicts (Movie Name, Description, Release Date, IMDB Rating)
- `texts` — just the descriptions, used as the searchable document corpus by all retrievers and the reranker

```
CSV row:
  Movie Name  : "Dhurandhar"
  Description : "Jaskirat Singh Rangi erases his identity and leaves his family..."
  Release Date: "2025-01-15"
  IMDB Rating : "9.4"

texts[0] = "Jaskirat Singh Rangi erases his identity and leaves his family..."
```

---

### Feature 2 — BM25 Retrieval (Stage 1-A)

BM25 (Best Match 25) is a keyword-based retrieval algorithm. It tokenises the corpus at startup and scores each document by counting how often query words appear in it, normalised by document length.

```
Query: "covert spy mission in Pakistan"

BM25 sees: ["covert", "spy", "mission", "in", "pakistan"]
Scores each of the 20 documents by word overlap.

Result: Documents containing "covert", "spy", or "mission" score high —
        even if those words appear in an unrelated context.
```

**Weakness:** `"Lakshya"` scores high because its description contains `"mission"` even though the movie is about army training, not espionage. BM25 cannot distinguish.

`top_k = 10` — cast a wide net. The reranker will narrow this down.

---

### Feature 3 — Dense Retrieval (Stage 1-B, Bi-Encoder)

Dense retrieval uses a **bi-encoder** (`all-MiniLM-L6-v2`): the query and each document are encoded **separately** into 384-dimensional vectors. Similarity is measured by cosine distance.

```
Query: "covert spy mission in Pakistan"
  → encoded → [0.23, -0.11, 0.87, ...]  (384 dims)

"Dhurandhar" description:
  → encoded → [0.25, -0.09, 0.84, ...]  (384 dims)

Cosine similarity → 0.51  (close in meaning space)
```

Document vectors are **pre-computed once at startup** and stored in memory. Only the query vector is computed at search time — making retrieval fast.

**Why needed alongside BM25:** Dense retrieval catches semantic matches — same meaning, different words. BM25 and dense retrieval cover different gaps, which is why they are combined in the hybrid pipeline (Feature 6).

---

### Feature 4 — Cross-Encoder Reranker (Stage 2)

The cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads the **query and each candidate document together** as a single combined input. This gives the model full attention across both texts simultaneously — far more accurate than a bi-encoder.

```
Input pair → reranker → relevance score 0–1

("covert spy mission in Pakistan",  "Jaskirat Singh Rangi erases his identity...") → 0.8972
("covert spy mission in Pakistan",  "An aimless young man finds his purpose...")    → 0.0041
```

**Key boundary:** The reranker is **NOT a retriever**. It cannot go and fetch new documents. It can only judge and reorder the documents passed to it. The first-stage retriever (BM25 or dense) must surface the right candidates for the reranker to work on.

**Why it is slower:** Unlike the bi-encoder, document vectors cannot be pre-computed — the model must run a fresh forward pass for every query–document pair.

The raw output (a logit) is converted to a 0–1 score via sigmoid. Relative ordering matters more than the absolute value.

---

### Feature 5 — keyword_and_reranking_search

This is the direct implementation of the book's `keyword_and_reranking_search` pattern (Figure 8-14). It is a two-stage pipeline:

```
Stage 1: BM25 keyword search  →  top-10 candidates  (cast wide net)
Stage 2: Cross-encoder reranker  →  top-3 final results  (narrow down hard)
```

```python
candidates = bm25.retrieve(query, top_k=10)
return reranker.rerank(query, candidates, top_n=3)
```

This pattern improves over BM25 alone because the reranker understands query intent. `"Lakshya"` (army training) scores `0.0041` from the reranker even though BM25 ranked it #2 — the reranker correctly identifies it as irrelevant to a spy query.

---

### Feature 6 — hybrid_reranking_search (Full 3-Stage Pipeline)

The full pipeline runs both retrievers in parallel, merges their results, and feeds the combined pool to the reranker.

```
User Query
     │
     ├────────────────────────────┐
     ▼                            ▼
 BM25 retrieval              Dense retrieval
 (keyword match)             (semantic match)
 top-10 candidates           top-10 candidates
     │                            │
     └─────────────┬──────────────┘
                   ▼
        Merge + deduplicate
        (~14 unique candidates)
                   │
                   ▼
      Cross-encoder Reranker
      Scores all 14 docs vs query
      Output: relevance score 0–1
                   │
                   ▼
         Top-3 Final Results
```

**Why three stages?**
- The reranker is slow — running it on all 20 documents for every query is expensive.
- BM25 + dense act as fast rough filters, narrowing from 20 → ~14 unique candidates.
- The reranker does precise ordering only on that small pool.

---

### Feature 7 — Precision@k

Precision@k answers: *"Of the top-k results returned, what fraction are actually relevant?"*

```
Precision@k = (relevant hits in top-k) / k

Example — BM25 Only (top-3 = [8, 19, 10], relevant = {0, 1, 2, 8, 10}):
  idx=8  → ✓ hit
  idx=19 → ✗ miss
  idx=10 → ✓ hit
  Precision@3 = 2/3 = 0.67

Example — BM25 + Reranker (top-3 = [0, 8, 10]):
  idx=0  → ✓ hit
  idx=8  → ✓ hit
  idx=10 → ✓ hit
  Precision@3 = 3/3 = 1.00
```

**Weakness:** Position-blind. A relevant result at rank #1 and at rank #3 contribute the same score. Average Precision (next) fixes this.

---

### Feature 8 — Average Precision (AP)

Average Precision rewards systems that rank relevant documents **higher** — not just in the top-k, but specifically at earlier positions.

```
Algorithm:
  For each position in the retrieved list:
    If the document at that position is relevant:
      Record Precision@that_position
  AP = sum of recorded precisions / total number of relevant documents

Example — BM25 + Reranker (retrieved = [0, 8, 10], relevant = {0,1,2,8,10}):
  Rank 1: idx=0  ✓  →  P@1 = 1/1 = 1.00  (recorded)
  Rank 2: idx=8  ✓  →  P@2 = 2/2 = 1.00  (recorded)
  Rank 3: idx=10 ✓  →  P@3 = 3/3 = 1.00  (recorded)
  AP = (1.00 + 1.00 + 1.00) / 5 = 0.60

Example — BM25 Only (retrieved = [8, 19, 10], relevant = {0,1,2,8,10}):
  Rank 1: idx=8  ✓  →  P@1 = 1/1 = 1.00  (recorded)
  Rank 2: idx=19 ✗  →  skipped
  Rank 3: idx=10 ✓  →  P@3 = 2/3 = 0.67  (recorded)
  AP = (1.00 + 0.67) / 5 = 0.33
```

---

### Feature 9 — Mean Average Precision (MAP)

AP works for a single query. MAP extends it across **all queries in the test suite** — producing one number to compare two search systems.

```
MAP = (AP_query1 + AP_query2 + AP_query3) / 3

Test Suite (3 queries with known relevant documents):
  Query 1: "covert spy mission in Pakistan"
    Relevant: {0, 1, 2, 8, 10}  ← Dhurandhar, Dhurandhar:Revenge, 16 Dec, Article 370, Madras Cafe

  Query 2: "soldiers battle war sacrifice"
    Relevant: {12, 13, 14, 15, 16, 17, 18, 19}  ← Uri, Shershaah, Border, Sam Bahadur, 1971, Major, Ghazi, Lakshya

  Query 3: "freedom fighter revolutionary assassination"
    Relevant: {3, 7, 11}  ← Sardar Udham, Sarfarosh, Bhagat Singh
```

---

## High Level Design (HLD)

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          main.py                                 │
│        Self-contained — runs all 9 features in sequence          │
│                                                                  │
│  Feature 1  →  Feature 2  →  Feature 3  →  Feature 4            │
│  (Data)        (BM25)        (Dense)        (Reranker)           │
│                                                                  │
│  Feature 5  →  Feature 6  →  Feature 7  →  Feature 8  → Feat 9  │
│  (BM25+Rank)   (Hybrid)      (P@k)          (AP)         (MAP)  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 1 — First-Stage Retrieval                 │
│                                                                  │
│  ┌────────────────┐         ┌────────────────────────────────┐  │
│  │ BM25Retriever  │         │ DenseRetriever (Bi-Encoder)    │  │
│  │                │         │                                │  │
│  │ rank_bm25      │         │ all-MiniLM-L6-v2               │  │
│  │ BM25Okapi      │         │ 384-dim embeddings             │  │
│  │                │         │ cosine similarity              │  │
│  │ Returns:       │         │                                │  │
│  │ top-k by       │         │ Returns:                       │  │
│  │ BM25 score     │         │ top-k by cosine score          │  │
│  └───────┬────────┘         └──────────────┬─────────────────┘  │
│          │                                 │                     │
│          └──────────────┬──────────────────┘                     │
│                         ▼                                        │
│              Merge + Deduplicate (~14 unique)                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2 — Cross-Encoder Reranker                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CrossEncoderReranker                                     │   │
│  │                                                          │   │
│  │ Model: cross-encoder/ms-marco-MiniLM-L-6-v2              │   │
│  │                                                          │   │
│  │ For each candidate:                                      │   │
│  │   input = (query, document)  →  model  →  raw logit      │   │
│  │   sigmoid(logit)  →  relevance score 0–1                 │   │
│  │                                                          │   │
│  │ Sort descending by relevance score → return top-n        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              EVALUATION — Retrieval Metrics                      │
│                                                                  │
│  Test Suite (3 queries + known relevant sets)                   │
│        │                                                         │
│        ▼                                                         │
│  Precision@k  →  relevant hits in top-k / k                     │
│        │                                                         │
│        ▼                                                         │
│  Average Precision  →  precision at relevant ranks, averaged    │
│        │                                                         │
│        ▼                                                         │
│  MAP  →  mean AP across all 3 queries  →  system score          │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Responsibility |
|---|---|---|
| **Data Loader** | `data_loader.py` / `main.py` | Reads CSV, returns movies list and descriptions list |
| **BM25Retriever** | `bm25_retriever.py` / `main.py` | Keyword-based Stage 1-A retrieval using `BM25Okapi` |
| **DenseRetriever** | `dense_retriever.py` / `main.py` | Bi-encoder Stage 1-B retrieval using `all-MiniLM-L6-v2` |
| **CrossEncoderReranker** | `reranker.py` / `main.py` | Stage 2 cross-encoder reranking using `ms-marco-MiniLM-L-6-v2` |
| **keyword_and_reranking_search** | `hybrid_search.py` / `main.py` | BM25 → Reranker two-stage pipeline (book pattern) |
| **hybrid_reranking_search** | `hybrid_search.py` / `main.py` | BM25 + Dense → merge → Reranker full pipeline |
| **Evaluation** | `evaluation.py` / `main.py` | Precision@k, AP, MAP with 3-query test suite |

### Data Flow

```
indian_movies.csv
 │
 ▼
20 movie descriptions  (texts list — the searchable corpus)
 │
 ├──────────────────────────────────────────┐
 ▼                                          ▼
BM25Okapi index (startup, keyword)    DenseRetriever index (startup, 384-dim vectors)
 │                                          │
 ▼  (query time)                            ▼  (query time)
BM25 scores per doc                   Cosine similarity per doc
 │                                          │
 └──────────────────┬───────────────────────┘
                    ▼
         Merge + deduplicate → candidate pool
                    │
                    ▼
      CrossEncoder.predict([(query, doc)] × N)
                    │
                    ▼
      sigmoid(logits) → relevance scores 0–1
                    │
                    ▼
      Sort descending → top-n final results
                    │
                    ▼
      Precision@k  →  AP per query  →  MAP across test suite
```

---

## Low Level Design (LLD)

### File Structure

```
reranking/
├── indian_movies.csv          ← input data (20 Indian movies)
├── reranking.docx             ← source study notes
├── requirements.txt           ← Python dependencies
├── README.md                  ← this file
├── main.py                    ← primary entry point — all 9 features self-contained
├── data_loader.py             ← CSV loading (used by module-based imports)
├── bm25_retriever.py          ← BM25Retriever class
├── dense_retriever.py         ← DenseRetriever class
├── reranker.py                ← CrossEncoderReranker class
├── hybrid_search.py           ← keyword_and_reranking_search + hybrid_reranking_search
└── evaluation.py              ← TEST_SUITE, precision_at_k, average_precision, MAP
```

> `main.py` is fully self-contained — all classes and functions are defined inline. The separate module files (`bm25_retriever.py`, `reranker.py`, etc.) are available as importable references for integration into other projects.

### Component Details

#### `main.py`

```
Purpose  : Self-contained entry point — runs all 9 features in one execution
           No imports from local modules needed — all code is inline

Config:
  DATA_FILE      = "indian_movies.csv"
  DEMO_QUERY     = "covert spy mission in Pakistan"
  TOP_K          = 10   (first-stage candidates — cast wide net)
  TOP_N          = 3    (final results after reranking — narrow down hard)
  DENSE_MODEL    = "all-MiniLM-L6-v2"
  RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

Execution sequence:
  1. Load CSV → movies, texts
  2. Build BM25Retriever(texts)
  3. Build DenseRetriever(texts)   ← downloads + encodes corpus at startup
  4. Build CrossEncoderReranker()  ← downloads cross-encoder at startup
  5. Run Features 1–9 in sequence with printed section headers

Usage:
  python main.py
```

#### `data_loader.py`

```
Function : load_movies()
Input    : reads indian_movies.csv from same directory
Output   : (movies: list[dict], texts: list[str])

movies → full CSV rows including Movie Name, IMDB Rating, Release Date
texts  → just the Description column — used as documents by all retrievers
```

#### `bm25_retriever.py`

```
Class    : BM25Retriever
Library  : rank_bm25 (BM25Okapi)

__init__(texts):
  Tokenises all descriptions (lowercase split)
  Builds BM25Okapi index at startup

retrieve(query, top_k=10) → list[dict]:
  Tokenises query
  Calls bm25.get_scores()
  Returns top_k dicts: {index, text, bm25_score}
```

#### `dense_retriever.py`

```
Class    : DenseRetriever
Model    : all-MiniLM-L6-v2 (sentence-transformers, bi-encoder)

__init__(texts):
  Loads SentenceTransformer model
  Encodes all 20 descriptions → shape (20, 384)
  L2-normalises embeddings → stored in memory

retrieve(query, top_k=10) → list[dict]:
  Encodes query → (384,) vector, normalised
  Dot product with stored doc embeddings = cosine similarity
  Returns top_k dicts: {index, text, dense_score}
```

#### `reranker.py`

```
Class    : CrossEncoderReranker
Model    : cross-encoder/ms-marco-MiniLM-L-6-v2

__init__():
  Loads CrossEncoder model

rerank(query, candidates, top_n=3) → list[dict]:
  Builds pairs: [(query, candidate_text) for each candidate]
  Calls model.predict(pairs) → raw logit per pair
  Applies sigmoid → relevance score 0–1
  Sorts descending, returns top_n
  Each result dict: original candidate fields + relevance_score
```

#### `hybrid_search.py`

```
Function 1: keyword_and_reranking_search(query, bm25, reranker, top_k, top_n)
  Stage 1: bm25.retrieve(query, top_k)     → candidates
  Stage 2: reranker.rerank(query, candidates, top_n)  → final results

Function 2: hybrid_reranking_search(query, bm25, dense, reranker, top_k, top_n)
  Stage 1A: bm25.retrieve(query, top_k)    → BM25 candidates
  Stage 1B: dense.retrieve(query, top_k)   → Dense candidates
  Stage 2:  Merge + deduplicate            → unique pool
  Stage 3:  reranker.rerank(query, pool, top_n)  → final results
```

#### `evaluation.py`

```
TEST_SUITE  : 3 queries with manually assigned relevant movie indices
  Query 1: "covert spy mission in Pakistan"      → {0, 1, 2, 8, 10}
  Query 2: "soldiers battle war sacrifice"       → {12, 13, 14, 15, 16, 17, 18, 19}
  Query 3: "freedom fighter revolutionary..."   → {3, 7, 11}

precision_at_k(retrieved, relevant, k)  → float
  hits in top-k / k

average_precision(retrieved, relevant)  → float
  precision at each relevant rank, averaged over total relevant count

mean_average_precision(pairs)           → float
  mean of AP scores across all queries

evaluate_system(search_fn, test_suite, top_n)  → dict
  Runs search_fn against all queries
  Returns {per_query: [...], MAP: float}
```

### Data Structures

**Input row (from CSV)**
```
{
  "Movie Name"  : "Dhurandhar",
  "Description" : "Jaskirat Singh Rangi erases his identity...",
  "Release Date": "2025-01-15",
  "IMDB Rating" : "9.4"
}
```

**BM25 retrieval result**
```
{
  "index"      : 0,
  "text"       : "Jaskirat Singh Rangi erases his identity...",
  "bm25_score" : 2.8442
}
```

**Dense retrieval result**
```
{
  "index"       : 2,
  "text"        : "A group of Indian intelligence officers...",
  "dense_score" : 0.5113
}
```

**Reranker result**
```
{
  "index"           : 0,
  "text"            : "Jaskirat Singh Rangi erases his identity...",
  "bm25_score"      : 2.8442,       ← carried over from BM25 stage
  "relevance_score" : 0.8972        ← added by reranker (sigmoid 0–1)
}
```

**Evaluation result (per system)**
```
{
  "per_query": [
    {
      "query"         : "covert spy mission in Pakistan",
      "retrieved"     : [0, 8, 10],
      "precision_at_k": 1.00,
      "ap"            : 0.600
    },
    ...
  ],
  "MAP": 0.469
}
```

---

## Dependencies

### System Requirements

| Requirement | Minimum Version | Purpose |
|---|---|---|
| **Python** | 3.10 or above | Runtime environment |
| **pip** | Latest | Package installer |
| **RAM** | 4 GB or above | Loading bi-encoder and cross-encoder into memory |
| **Disk Space** | ~200 MB free | all-MiniLM-L6-v2 (~90 MB) + ms-marco-MiniLM-L-6-v2 (~80 MB) |
| **Internet Connection** | Required on first run | Download models from HuggingFace |

> After the first run, models are cached locally at `~/.cache/huggingface/hub/`. Internet is not needed for subsequent runs.

### Python Libraries

Installed via `pip install -r requirements.txt`:

| Library | Version | Purpose |
|---|---|---|
| `rank-bm25` | >=0.2.2 | BM25Okapi keyword retrieval index |
| `sentence-transformers` | >=2.7.0 | Bi-encoder (Stage 1-B) and cross-encoder reranker (Stage 2) |
| `numpy` | >=1.24.0 | Cosine similarity computation, sigmoid function |

### Models Used

| Feature | Model | Type | Size | Purpose |
|---|---|---|---|---|
| 3, 6 | `all-MiniLM-L6-v2` | Bi-encoder | ~90 MB | Dense retrieval — encodes query and docs separately into 384-dim vectors |
| 4, 5, 6 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder | ~80 MB | Reranking — reads query + doc together, outputs relevance score 0–1 |

### What Gets Downloaded on First Run

```
Feature 3 — all-MiniLM-L6-v2  (bi-encoder)
  └── downloads from HuggingFace
        ├── config.json
        ├── tokenizer files
        └── model weights (~90 MB)

Feature 4 — cross-encoder/ms-marco-MiniLM-L-6-v2
  └── downloads from HuggingFace
        ├── config.json
        ├── tokenizer files
        └── model weights (~80 MB)
```

> No API key or account required. Both models are open-source and free to use.
> `rank-bm25` requires no model download — it is a pure Python algorithm.

---

## How to Run

### Step 1 — Navigate to the project directory

```bash
cd /path/to/reranking
```

### Step 2 — Create a virtual environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run

```bash
python main.py
```

On first run, two models are downloaded from HuggingFace (~170 MB total). Once ready you will see the features execute in sequence, each with a clear section header. The program runs fully automatically — no user input required.

---

## Sample Output

### Startup

```
══════════════════════════════════════════════════════════════
  FEATURE 1 — Data Loading
══════════════════════════════════════════════════════════════

  File    : indian_movies.csv
  Records : 20
  Columns : Movie Name, Description, Release Date, IMDB Rating

  Sample documents (used as search corpus):
    [0] Dhurandhar — Jaskirat Singh Rangi erases his identity and leaves his fami...
    [1] Dhurandhar: The Revenge — The protagonist continues his deep-cover mission...
    [2] 16 December — A group of Indian intelligence officers use high-tech equipm...
    ...
```

---

### Feature 2 — BM25 Retrieval

```
══════════════════════════════════════════════════════════════
  FEATURE 2 — BM25 Retrieval  (keyword-based, Stage 1-A)
══════════════════════════════════════════════════════════════

  Query: "covert spy mission in Pakistan"
  (Scores documents by word overlap — fast but meaning-blind)

  ──────────────────────────────────────────────────────────
  BM25 top-3
  ──────────────────────────────────────────────────────────
  #1  [2.8442]  Article 370
       A high-stakes political and intelligence thriller exploring the covert o...
  #2  [2.6470]  Lakshya
       An aimless young man finds his purpose and becomes a dedicated army offi...
  #3  [0.1938]  Madras Cafe
       An Indian intelligence officer is caught in a web of international consp...
```

> Note: `Lakshya` ranks #2 because its description contains the word `"mission"` — a classic BM25 false positive. The reranker corrects this.

---

### Feature 4 — Cross-Encoder Reranker

```
══════════════════════════════════════════════════════════════
  FEATURE 4 — Cross-Encoder Reranker  (Stage 2)
══════════════════════════════════════════════════════════════

  Query: "covert spy mission in Pakistan"
  Input : 10 BM25 candidates  →  reranker reads (query + each doc) together
  Output: top-3 sorted by relevance score 0–1

  ──────────────────────────────────────────────────────────
  Cross-Encoder top-3  (from 10 BM25 candidates)
  ──────────────────────────────────────────────────────────
  #1  [0.8972]  Dhurandhar
       Jaskirat Singh Rangi erases his identity and leaves his family to live a...
  #2  [0.7533]  Article 370
       A high-stakes political and intelligence thriller exploring the covert o...
  #3  [0.0041]  Madras Cafe
       An Indian intelligence officer is caught in a web of international consp...
```

> `Dhurandhar` jumps to #1. `Lakshya` (army training) is pushed out of top-3 entirely — scored near 0 by the reranker because it understands the query is about espionage, not military drills.

---

### Feature 6 — Hybrid Pipeline

```
══════════════════════════════════════════════════════════════
  FEATURE 6 — hybrid_reranking_search  (Full 3-stage pipeline)
══════════════════════════════════════════════════════════════

  Query: "covert spy mission in Pakistan"
  Stage 1A : BM25   → 10 candidates
  Stage 1B : Dense  → 10 candidates
  Stage 2  : Merge + deduplicate → 14 unique candidates
  Stage 3  : Reranker → top-3 final results

  ──────────────────────────────────────────────────────────
  Hybrid (BM25 + Dense) → Reranker
  ──────────────────────────────────────────────────────────
  #1  [0.8972]  Dhurandhar
  #2  [0.7533]  Article 370
  #3  [0.0131]  The Ghazi Attack
```

---

### Feature 8 — Average Precision Walkthrough

```
══════════════════════════════════════════════════════════════
  FEATURE 8 — Average Precision  (rewards early relevant hits)
══════════════════════════════════════════════════════════════

  Query: "covert spy mission in Pakistan"  |  Relevant: [0, 1, 2, 8, 10]

  Walkthrough — BM25 + Reranker
  ──────────────────────────────────────────────────────────
  Retrieved : [0, 8, 10]
  Relevant  : [0, 1, 2, 8, 10]
    Rank 1: idx=0  ✓  →  P@1 = 1/1 = 1.00  (recorded)
    Rank 2: idx=8  ✓  →  P@2 = 2/2 = 1.00  (recorded)
    Rank 3: idx=10 ✓  →  P@3 = 3/3 = 1.00  (recorded)
  AP = (1.00 + 1.00 + 1.00) / 5 = 0.600
```

---

### Feature 9 — MAP Final Comparison

```
══════════════════════════════════════════════════════════════
  SUMMARY — MAP Comparison across all 3 pipelines
══════════════════════════════════════════════════════════════

  BM25 Only               MAP=0.236  ████████
  BM25 + Reranker         MAP=0.469  ████████████████
  Hybrid + Reranker       MAP=0.444  ███████████████

  Best system: BM25 + Reranker
```

---

## Concepts from Study Notes → Features

| Study Note / Book Concept | Feature |
|---|---|
| BM25 matches words not meaning — keyword false positives | 2 |
| First-stage retrieval: cast a wide net (top_k=10) | 2, 3 |
| Bi-encoder: query and docs encoded separately, cosine similarity | 3 |
| Hybrid search: BM25 + dense run in parallel, pools combined | 6 |
| Cross-encoder: reads query + document together as one input | 4 |
| Reranker is NOT a retriever — can only reorder what it receives | 4, 5, 6 |
| Relevance score 0–1 via sigmoid — relative ordering matters | 4, 5, 6 |
| keyword_and_reranking_search pattern (Figure 8-14) | 5 |
| Two-stage pipeline: retrieval → reranking (Figure 8-14) | 5, 6 |
| Test suite: queries with known relevant documents (Figure 8-16) | 7, 8, 9 |
| Precision@k: relevant hits in top-k / k (Figure 8-21) | 7 |
| Precision@k is position-blind — AP fixes this (Figure 8-22) | 7, 8 |
| Average Precision: precision at relevant ranks, averaged (Figure 8-20) | 8 |
| MAP: mean AP across all queries — system benchmark (Figure 8-23) | 9 |

---

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'rank_bm25'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| Slow startup (30–60 seconds on first run) | Models downloading from HuggingFace (~170 MB) | Wait — downloads happen only once, then cached |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| BM25 returns unexpected results | Stop words like "in", "the" inflating scores | Expected BM25 behaviour — the reranker corrects these in stage 2 |
| MAP scores differ between runs | Model randomness or floating point variation | Scores are deterministic — check if correct model versions are installed |
