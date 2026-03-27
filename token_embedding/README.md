# Token Embeddings — Hands-on Learning Project

## Project Description

This project is a hands-on, runnable implementation of the concepts covered in the *Token Embeddings* study notes. It builds from first principles — starting with how raw text becomes numbers — and progresses through four progressive scripts that each demonstrate a distinct layer of how modern language models understand language.

The dataset used throughout is `indian_movies.csv`, a collection of 20 Indian movies with names, descriptions, release dates, and IMDB ratings. Every concept from the study notes is grounded in this real dataset so the output is always meaningful and relatable.

The project is structured as a learning journey, not a production system. Each script is self-contained, heavily commented, and designed to be read alongside the study notes.

---

## Problem Statement

Language models cannot work with raw text. Every word, subword, or character must be converted into numbers before any computation can happen. But the conversion must also carry **meaning** — simply assigning `1` to "spy" and `2` to "soldier" would tell the model nothing about their relationship.

This raises three interconnected problems that this project addresses:

| Problem | What It Means | How This Project Addresses It |
|---|---|---|
| Text → numbers | Models need numeric input, not strings | Script 01 shows how BERT's embedding matrix converts token IDs to vectors |
| Static meaning | The word "spy" means different things in different sentences | Script 02 shows how BERT produces different vectors for the same word depending on context |
| Token → sentence | A sentence has many tokens but tasks often need one single vector | Script 03 shows mean-pooling and produces one 384-dim vector per movie description |
| High-dimensional space | Embeddings live in 384 or 768 dimensions — impossible to visualise directly | Script 04 uses PCA to project down to 2-D and plots the geometry of meaning |

---

## Learning Objectives

By running all four scripts and reading their output, you will be able to answer:

- What is the embedding matrix and how is it structured?
- What is a token ID and how does tokenisation work?
- What does "768 dimensions" actually mean geometrically?
- Why do static embeddings fail for words like "spy" or "bank"?
- How does BERT produce contextual vectors and what changes between sentences?
- What are `[CLS]` and `[SEP]` tokens and why do they appear in the output shape?
- How is a sentence converted into a single vector using mean-pooling?
- What is cosine similarity and why is it the right measure for comparing embeddings?
- What does the embedding space look like and do similar movies actually cluster together?

---

## How We Are Solving It

The project is split into four independent, progressive scripts. Each script builds on the concepts introduced in the previous one.

### Script 01 — The Embedding Matrix

**Goal:** Understand that embeddings are not magic — they are a lookup table.

A BERT model contains an embedding matrix of shape `[30,522 × 768]`. The number 30,522 is the vocabulary size — every token the tokeniser knows. The number 768 is the embedding dimension — the number of values in each token's vector.

When a sentence arrives, it is first tokenised into a list of integer IDs. Then each ID is used to look up its row in the matrix. The result is a list of 768-dimensional vectors — one per token. No computation has happened yet. This is a pure table lookup.

```
Input:  "Sardar Udham"

Step 1 — Tokenise:
  "Sardar Udham" → [CLS] sar ##dar ud ##ham [SEP]
                 → [101, 18906, 7662, 20904, 3511, 102]

Step 2 — Lookup each ID in the matrix:
  101   → [ 0.014, -0.026, -0.024, ...]   ← row 101 of the matrix
  18906 → [-0.016, -0.050, -0.021, ...]   ← row 18906
  ...
```

This is the **static embedding** — the raw vector assigned to a token before any context is applied.

---

### Script 02 — Static vs Contextual Embeddings

**Goal:** Understand why static embeddings are insufficient and how BERT fixes them.

The word `"spy"` appears in multiple movie descriptions with very different surrounding words:

```
Sentence A: "he operates as a spy knowing he will die at the hands of terrorists"
Sentence B: "intelligence spy thriller about recovering a stolen nuclear bomb"
```

The static vector for `"spy"` is identical in both sentences — it is just row 8,645 of the matrix, read as-is. Cosine similarity between the two static vectors = **1.0**. The model sees no difference.

After passing both sentences through BERT's transformer layers, the same token gets shaped by everything around it. Every token attends to every other token. The output vector for `"spy"` is now different in each sentence.

```
Static  'spy' vs 'spy'        → cosine similarity: 1.0000  (always — same row)
BERT    'spy' in sentence A   → [0.979, -0.090, 0.379, ...]
BERT    'spy' in sentence B   → [1.050, -0.313, 0.133, ...]
                              → cosine similarity: 0.7148  (context changed it)
```

This script also demonstrates:
- The output shape `[1, 16, 768]` — what batch, seq_len, and dims mean
- How `[CLS]` and `[SEP]` are automatically added and take up positions in the output

---

### Script 03 — Text Embeddings and Semantic Search

**Goal:** Collapse a full movie description (many tokens) into one vector, then use it to find similar movies.

Token embeddings give one vector per token. But semantic search requires one vector per document. The standard approach is **mean-pooling** — average all token vectors into a single vector.

```
"Best"  → [0.23, 0.81, ...]
"movie" → [0.54, 0.12, ...]
"ever"  → [0.11, 0.94, ...]
"!"     → [0.33, 0.44, ...]
────────────────────────────
Average → [0.30, 0.57, ...]   ← one 384-dim vector for the whole sentence
```

All 20 movie descriptions are encoded this way. When a user types a query, the query is encoded with the same model and the same pooling. Cosine similarity is computed between the query vector and every movie vector. Movies are ranked by score.

This works because both the movies and the query are embedded by the **same model**, so semantically similar texts land close together in the 384-dimensional space — even with completely different words.

---

### Script 04 — Visualising the Embedding Space

**Goal:** See the geometry of meaning — prove that similar movies cluster together without any genre labels.

Human eyes cannot see 384 dimensions. PCA (Principal Component Analysis) projects the 384-dim movie vectors down to 2-D by finding the two directions of maximum variance. Each movie becomes a point on a 2-D plot.

The key observation: spy/espionage movies cluster in one region, war movies in another, freedom fighters in another — entirely on their own, from the raw text of descriptions.

```
384-D embedding vectors
        │
        ▼  PCA (finds 2 axes of max variance)
        │
2-D coordinates  →  scatter plot
```

The script also prints each movie's **nearest neighbour** in embedding space — which movie is geometrically closest in meaning.

---

## High Level Design (HLD)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCRIPT 01 & 02                           │
│                    (Token-Level Analysis)                        │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │Raw Text  │───▶│  Tokeniser   │───▶│   Embedding Matrix     │ │
│  │(movie    │    │(BertTokenizer│    │   [30,522 × 768]       │ │
│  │ title)   │    │)             │    │   static lookup        │ │
│  └──────────┘    └──────────────┘    └────────────────────────┘ │
│                                                 │               │
│                                                 ▼               │
│                                      ┌─────────────────────┐   │
│                                      │  BERT Transformer   │   │
│                                      │  (12 attention      │   │
│                                      │   layers)           │   │
│                                      └──────────┬──────────┘   │
│                                                 │               │
│                                                 ▼               │
│                                      ┌─────────────────────┐   │
│                                      │ Contextual Vectors  │   │
│                                      │ [1 × tokens × 768]  │   │
│                                      └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        SCRIPT 03 & 04                           │
│                   (Sentence-Level Analysis)                      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ indian_      │───▶│ SentenceTransfor-│───▶│ Text Vectors  │  │
│  │ movies.csv   │    │ mer              │    │ [20 × 384]    │  │
│  │ (20 movies)  │    │ (MiniLM-L6-v2)   │    │               │  │
│  └──────────────┘    │  + Mean Pooling  │    └──────┬────────┘  │
│                      └──────────────────┘           │           │
│                                                     │           │
│  ┌──────────┐    ┌──────────────────┐               │           │
│  │ User     │───▶│ Same Model       │───▶ Query     │           │
│  │ Query    │    │ + Mean Pooling   │     Vector    │           │
│  └──────────┘    └──────────────────┘     [1 × 384] │           │
│                                                     │           │
│                                          Cosine     ▼           │
│                                          Similarity / PCA       │
│                                                     │           │
│                                                     ▼           │
│                                          ┌─────────────────┐   │
│                                          │ Top-K Results / │   │
│                                          │ 2-D Cluster Plot│   │
│                                          └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Script | Responsibility |
|---|---|---|
| **BertTokenizer** | 01, 02 | Converts raw text to token IDs using WordPiece tokenisation |
| **Embedding Matrix** | 01, 02 | 30,522 × 768 table — maps each token ID to a static 768-dim vector |
| **BERT Transformer** | 02 | 12 attention layers that reshape each token's vector based on full sentence context |
| **SentenceTransformer** | 03, 04 | Encodes full sentences into single 384-dim vectors via mean-pooling |
| **Cosine Similarity** | 03 | Measures angle between two vectors — 1.0 = same direction = same meaning |
| **PCA** | 04 | Reduces 384-D → 2-D for visualisation while preserving maximum variance |

### Data Flow

```
Raw Text (movie title / description / query)
 │
 ▼
Tokenisation  →  Token IDs  →  [101, 18906, 7662, 20904, 3511, 102]
 │
 ▼
Static Embedding Lookup  →  [N tokens × 768 dims]  (matrix row reads)
 │
 ▼  (Script 01 stops here — pure matrix lookup)
 │
 ▼
BERT Transformer (attention over all tokens)
 │
 ▼
Contextual Vectors  →  [1 × N tokens × 768 dims]  (Script 02 stops here)
 │
 ▼
Mean Pooling  →  [1 × 384 dims]  (one vector per sentence)
 │
 ▼
Cosine Similarity  →  Ranked list of matching movies  (Script 03)
       or
PCA 2-D Projection  →  Scatter plot of embedding space  (Script 04)
```

---

## Low Level Design (LLD)

### File Structure

```
token_embedding/
├── indian_movies.csv              ← input data (20 Indian movies)
├── Token_Embeddings.docx          ← source study notes
├── requirements.txt               ← Python dependencies
├── README.md                      ← this file
├── embedding_clusters.png         ← generated by Script 04
└── src/
    ├── __init__.py
    ├── 01_embedding_basics.py     ← embedding matrix, tokenisation, dims
    ├── 02_contextual_embeddings.py← static vs contextual, BERT pipeline
    ├── 03_semantic_search.py      ← text embeddings, cosine search (interactive)
    └── 04_visualize_clusters.py   ← PCA plot of all 20 movie embeddings
```

### Component Details

#### `01_embedding_basics.py`

```
Purpose  : Demonstrate that embeddings are a table lookup, not magic
Model    : bert-base-uncased (BertTokenizer + BertModel)

Steps:
  1. Load BertTokenizer and BertModel
  2. Extract embedding_matrix = model.embeddings.word_embeddings.weight
     Shape: [30,522 × 768]
  3. Tokenise "Sardar Udham" → token IDs
  4. For each token ID, read its row from the matrix directly
  5. Print the first 6 dimensions of each token's static vector
  6. Explain 768 dimensions using the 1-D / 2-D / 768-D analogy

Key output:
  Token IDs   : [101, 18906, 7662, 20904, 3511, 102]
  Tokens      : ['[CLS]', 'sar', '##dar', 'ud', '##ham', '[SEP]']
  Static vector for each token (first 6 dims shown)
  Shape of one vector: (768,)
```

#### `02_contextual_embeddings.py`

```
Purpose  : Show that BERT produces different vectors for the same word
           in different sentence contexts
Model    : bert-base-uncased

Steps:
  1. Define two sentences, both containing the word "spy"
  2. Look up "spy" static vector directly from embedding matrix (same both times)
  3. Run each sentence through BERT's full transformer
  4. Extract the output vector for "spy" from each sentence's output
  5. Compute cosine similarity: static pair vs contextual pair
  6. Demonstrate [CLS] and [SEP] tokens and explain output shape

Key output:
  Static   'spy' vs 'spy'   → cosine: 1.0000  (always — row read)
  BERT     'spy' in sent A  → different numbers
  BERT     'spy' in sent B  → different numbers
  Contextual similarity     → ~0.71  (context changed the vector)
  Output shape: torch.Size([1, 16, 768])
```

#### `03_semantic_search.py`

```
Purpose  : Interactive semantic search over all 20 movies
Model    : sentence-transformers/all-MiniLM-L6-v2

Startup:
  1. Load all-MiniLM-L6-v2
  2. Read all 20 movie descriptions from indian_movies.csv
  3. Encode each description → 384-dim vector  (shape: [20, 384])

Search loop (per query):
  1. Encode user query → 384-dim query vector
  2. Compute cosine_similarity(query_vec, movie_vec) for all 20 movies
  3. Sort by score descending
  4. Return top-3 with rank, score, IMDB rating, and description snippet

Function signatures:
  load_movies(path)                       → list[dict]
  cosine_similarity(a, b)                 → float
  search(query, movies, embeddings, model, top_k) → list[(movie, score)]
```

#### `04_visualize_clusters.py`

```
Purpose  : Visualise that similar movies cluster together in embedding space
Model    : sentence-transformers/all-MiniLM-L6-v2

Steps:
  1. Encode all 20 descriptions → embeddings [20 × 384]
  2. PCA: reduce [20 × 384] → [20 × 2]
  3. Plot each movie as a 2-D point, coloured by genre group
  4. Annotate each point with the movie name
  5. Print nearest neighbour table (closest movie in embedding space)
  6. Save plot to embedding_clusters.png

Genre groups (used for colour only — NOT given to the model):
  Spy / Espionage   → red
  War / Military    → blue
  Freedom Fighter   → green
  Civilian / Social → orange

Key output:
  embedding_clusters.png
  Nearest neighbour table (20 rows)
  PCA variance explained: PC1=X%  PC2=Y%  Total=Z%
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

**Static embedding lookup result (Script 01)**
```
token_id  : 8645
token     : "spy"
vector    : [ 0.0244,  0.0179, -0.0119, -0.0609,  0.0531, ...] (768 values)
shape     : (768,)
```

**BERT contextual output (Script 02)**
```
output.last_hidden_state → torch.Size([1, 16, 768])
                                        ↑   ↑   ↑
                               batch=1  tokens dims=768

Contextual vector for 'spy' in sentence A:
  [ 0.9797, -0.0904,  0.3792, -0.2706,  0.5664, ...]  (768 values)
```

**Text embedding (Script 03, 04)**
```
"Dhurandhar. Jaskirat Singh Rangi erases his identity..."
  → mean_pool(all token vectors)
  → [0.23, -0.11, 0.87, ...]   (384 values)
  → shape: (384,)
```

**Search result (Script 03)**
```
{
  "Movie Name"  : "Lakshya",
  "Description" : "An aimless young man finds his purpose...",
  "Release Date": "2004-06-18",
  "IMDB Rating" : "8.0",
  "score"       : 0.8231     ← cosine similarity with query
}
```

---

## Dependencies

### System Requirements

| Requirement | Minimum Version | Purpose |
|---|---|---|
| **Python** | 3.10 or above | Runtime environment |
| **pip** | Latest | Package installer |
| **RAM** | 4 GB or above | Loading BERT or MiniLM into memory |
| **Disk Space** | ~600 MB free | bert-base-uncased (~440 MB) + all-MiniLM-L6-v2 (~90 MB) |
| **Internet Connection** | Required on first run | Download models from HuggingFace |

> After the first run, models are cached locally at `~/.cache/huggingface/hub/`. Internet is not needed for subsequent runs.

### Python Libraries

Installed via `pip install -r requirements.txt`:

| Library | Purpose |
|---|---|
| `torch` | Run BERT forward pass, tensor operations |
| `transformers` | BertTokenizer and BertModel for Scripts 01 and 02 |
| `sentence-transformers` | all-MiniLM-L6-v2 for Scripts 03 and 04 |
| `numpy` | Vector maths, cosine similarity, PCA input |
| `scikit-learn` | PCA dimensionality reduction in Script 04 |
| `matplotlib` | Scatter plot of embedding space in Script 04 |

### Models Used

| Script | Model | Dim | Size | Purpose |
|---|---|---|---|---|
| 01, 02 | `bert-base-uncased` | 768 | ~440 MB | Token-level embeddings, static and contextual |
| 03, 04 | `all-MiniLM-L6-v2` | 384 | ~90 MB | Sentence-level embeddings, semantic search |

### What Gets Downloaded on First Run

```
Scripts 01, 02 — bert-base-uncased
  └── downloads from HuggingFace
        ├── config.json
        ├── vocab.txt  (30,522 tokens)
        ├── tokenizer_config.json
        └── pytorch_model.bin  (~440 MB)

Scripts 03, 04 — all-MiniLM-L6-v2
  └── downloads from sentence-transformers
        ├── config.json
        ├── tokenizer files
        └── model weights  (~90 MB)
```

> No API key or account is required. Both models are open-source and free to use.

---

## How to Run

### Step 1 — Navigate to the project directory

```bash
cd /path/to/token_embedding
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

### Step 4 — Run the scripts in order

Each script is independent. Run them in order for the full learning progression.

```bash
python src/01_embedding_basics.py
python src/02_contextual_embeddings.py
python src/03_semantic_search.py
python src/04_visualize_clusters.py
```

> Scripts 01 and 02 download `bert-base-uncased` (~440 MB) on first run.
> Scripts 03 and 04 download `all-MiniLM-L6-v2` (~90 MB) on first run.
> Both are cached and reused on every subsequent run.

---

## Sample Output

### Script 01

```
Loading BERT tokenizer and model...

── Embedding Matrix ─────────────────────────────────────────────────
  Vocabulary size : 30,522 tokens
  Embedding dims  : 768
  Total params    : 23,440,896
  Memory (float32): ~93.8 MB

── Tokenisation: 'Sardar Udham' ──────────────────────────────────────────
  Token IDs : [101, 18906, 7662, 20904, 3511, 102]
  Tokens    : ['[CLS]', 'sar', '##dar', 'ud', '##ham', '[SEP]']

  ID      Token          First 6 dims of static vector
  ─────────────────────────────────────────────────────
  101     [CLS]           [ 0.014 -0.026 -0.024 -0.008  0.009 -0.008]
  18906   sar             [-0.016 -0.050 -0.021  0.012 -0.047 -0.009]
  7662    ##dar           [ 0.019 -0.033 -0.030 -0.111 -0.058 -0.022]
  20904   ud              [ 0.010 -0.082 -0.041 -0.040 -0.021  0.014]
  3511    ##ham           [-0.078  0.030 -0.074 -0.111 -0.037 -0.040]
  102     [SEP]           [-0.015 -0.010  0.006 -0.009 -0.019 -0.025]
```

### Script 02

```
── Step 3 — Cosine Similarity Comparison ────────────────────────────
  Static   'spy' vs 'spy'  (same matrix row)   : 1.0000  ← always 1.0
  Contextual A vs B  (different sentence context) : 0.7148  ← context matters
```

### Script 03

```
============================================================
  SEMANTIC MOVIE SEARCH
  Find movies by meaning, not keywords.
============================================================

Query: spy who sacrifices everything for the nation

  Top 3 matches for: 'spy who sacrifices everything for the nation'
  #   Score    IMDB   Movie
  ─── ─────── ─────  ───────────────────────────────────
  1   0.8611   9.4    Dhurandhar
      ↳ Jaskirat Singh Rangi erases his identity and leaves his family to live as a...
  2   0.8287   9.6    Dhurandhar: The Revenge
      ↳ The protagonist continues his deep-cover mission, accepting a nameless, lon...
  3   0.7543   7.7    Madras Cafe
      ↳ An Indian intelligence officer is caught in a web of international conspira...
```

### Script 04

```
  PCA: reduced 384-D → 2-D
  Variance explained: PC1=28.3%  PC2=14.1%  Total=42.4%

── Nearest Neighbour in Embedding Space ─────────────────────────────
  Movie                                    Nearest neighbour
  ──────────────────────────────────────── ───────────────────────────────────
  Dhurandhar                               Dhurandhar: The Revenge
  Uri: The Surgical Strike                 Shershaah
  Sardar Udham                             The Legend of Bhagat Singh
  Swades                                   Airlift
  ...

  Plot saved → embedding_clusters.png
```

---

## Concepts from Study Notes → Scripts

| Study Note Topic | Script |
|---|---|
| Embedding is a numeric vector assigned to a token | 01 |
| Embedding matrix: one row per vocabulary token | 01 |
| Vocabulary size and embedding dim (GPT-3 example) | 01 |
| Token IDs and tokenisation pipeline | 01 |
| What "768 dimensions" means geometrically | 01 |
| Tokeniser and model are permanently linked | 01 |
| Static embeddings: same token → same vector always | 01, 02 |
| Contextual embeddings: context reshapes each token's vector | 02 |
| Full pipeline: raw text → static → BERT → contextual | 02 |
| Special tokens [CLS] and [SEP] | 02 |
| Output shape [batch, tokens, dims] explained | 02 |
| Text embedding: one vector for a whole sentence | 03 |
| Mean-pooling to collapse token vectors | 03 |
| Cosine similarity for semantic matching | 03 |
| Embedding space: similar meaning = nearby point | 03, 04 |
| Embedding space vs embedding matrix (different things) | 04 |
| Similar words cluster together after training | 04 |
| Embedding matrix is frozen during inference | 01, 02 |
