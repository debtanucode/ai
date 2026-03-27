# Indian Movie Semantic Search using Dense Retrieval

## Project Description

This project demonstrates how to build a semantic search engine for Indian movies using the concept of **dense retrieval**. Instead of matching keywords, the system understands the **meaning** behind a user's query and retrieves the most relevant movies based on semantic similarity.

The project is built as a hands-on implementation of the concepts covered in *Hands-On Large Language Models* — Chapter 8: Semantic Search and Retrieval-Augmented Generation.

---

## Problem Statement

Traditional keyword-based search has a fundamental limitation — it only matches **exact words**, not meaning.

For example, if a user searches:
> *"a movie about a soldier sacrificing his life for the country"*

A keyword search would only return results containing those exact words. It would **miss** movies like *Lakshya* or *Major* whose descriptions talk about *"army officer"*, *"mission"*, *"heroic death"* — same meaning, different words.

**This project solves that problem** by converting movie descriptions into dense vectors (embeddings) that capture semantic meaning. When a user queries, the query is also embedded and the system finds movies whose vectors are **closest** in meaning — regardless of the exact words used.

| Challenge | Solution |
|---|---|
| Keyword search misses semantically similar content | Dense embeddings capture meaning, not just words |
| No structured way to search a movie dataset | FAISS index enables fast nearest neighbour search |
| Hard to find movies by theme or mood | Query embedding maps theme/mood to vector space |

---

## How We Are Solving It

The solution follows a two-phase approach — **indexing** and **searching**.

### Phase 1 — Indexing (done once at startup)

**1. Load the data**
The movie dataset is loaded from a CSV file. Each movie's name and description are combined into a single text chunk — one chunk per movie.

```
"Dhurandhar. Jaskirat Singh Rangi erases his identity and leaves his family..."
```

**2. Embed the chunks**
Each text chunk is passed through a pre-trained sentence embedding model (`all-MiniLM-L6-v2`). The model converts each chunk into a **384-dimensional vector** that captures its semantic meaning.

```
"Dhurandhar. Jaskirat Singh..."  →  [0.23, -0.11, 0.87, ...]  (384 numbers)
```

**3. Build the vector index**
All movie vectors are stored in a **FAISS index** — a structure optimised for fast similarity search across large collections of vectors.

### Phase 2 — Searching (done on every query)

**1. Embed the query**
The user's query is passed through the **same embedding model**, producing a query vector in the same 384-dimensional space.

**2. Find nearest neighbours**
FAISS computes the **L2 distance** between the query vector and every movie vector. Lower distance = higher similarity.

**3. Return results**
The top-k closest movies are returned with their name, IMDB rating, description, and distance score.

### Why This Works

Because both the movies and the query are embedded by the **same model**, semantically similar texts end up **close together in vector space** — even if they share no common words. This is the core principle of dense retrieval.

---

## High Level Design (HLD)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INDEXING PHASE                       │
│                                                         │
│  ┌──────────┐    ┌────────────┐    ┌─────────────────┐  │
│  │ CSV File │───▶│ Data Loader│───▶│ Embedding Model │  │
│  └──────────┘    └────────────┘    └────────┬────────┘  │
│                                             │            │
│                                             ▼            │
│                                    ┌─────────────────┐  │
│                                    │   FAISS Index   │  │
│                                    └─────────────────┘  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     SEARCH PHASE                         │
│                                                         │
│  ┌───────────┐    ┌─────────────────┐    ┌───────────┐  │
│  │ User Query│───▶│ Embedding Model │───▶│   FAISS   │  │
│  └───────────┘    └─────────────────┘    │  Search   │  │
│                                          └─────┬─────┘  │
│                                                │         │
│                                                ▼         │
│                                       ┌───────────────┐ │
│                                       │  Top-K Movies │ │
│                                       └───────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Components

| Component | Responsibility |
|---|---|
| **Data Loader** | Reads CSV, prepares text chunks |
| **Embedding Model** | Converts text to dense vectors |
| **FAISS Index** | Stores vectors, performs similarity search |
| **Search Engine** | Orchestrates all components |
| **CLI (main.py)** | User interaction layer |

### Data Flow

```
CSV
 │
 ▼
Text Chunks (Movie Name + Description)
 │
 ▼
Dense Vectors [384 dimensions]
 │
 ▼
FAISS Index
 │
 ◀── Query Vector (from user query)
 │
 ▼
Top-K Similar Movies
```

---

## Low Level Design (LLD)

### File Structure

```
dense_retrieval/
├── indian_movies.csv          ← input data
├── requirements.txt           ← dependencies
├── README.md                  ← documentation
├── main.py                    ← CLI entry point
└── src/
    ├── __init__.py
    ├── data_loader.py         ← CSV loading & chunking
    ├── embedder.py            ← vector generation
    ├── search_index.py        ← FAISS index management
    └── search.py              ← orchestration layer
```

### Component Details

#### `data_loader.py`

```
Function : load_movies(csv_path)
Input    : path to CSV file
Process  : read CSV → detect name column → combine name + description
Output   : movie_chunks (list[str]), movie_metadata (list[dict])

movie_chunks   → ["Dhurandhar. Jaskirat Singh...", "Lakshya. An aimless young man...", ...]
movie_metadata → [{"Movie Name": "Dhurandhar", "IMDB Rating": 9.4, ...}, ...]
```

#### `embedder.py`

```
Class  : Embedder
Model  : all-MiniLM-L6-v2 (sentence-transformers)
Output : float32 numpy arrays

Methods:
  embed(texts)       → np.ndarray shape (N, 384)   ← for indexing all movies
  embed_query(query) → np.ndarray shape (1, 384)   ← for a single search query
```

#### `search_index.py`

```
Class  : SearchIndex
Index  : FAISS IndexFlatL2 (exact L2 distance)

Methods:
  build(embeddings)              → stores N vectors in FAISS index
  search(query_embedding, top_k) → returns (distances, indices)
                                   distances shape : (1, top_k)
                                   indices shape   : (1, top_k)
```

#### `search.py`

```
Class  : MovieSearchEngine
Role   : Orchestrates data_loader → embedder → search_index

Init:
  1. load_movies()       → movie_chunks, movie_metadata
  2. embedder.embed()    → chunk_embeddings  (N, 384)
  3. index.build()       → FAISS index ready

search(query, top_k):
  1. embedder.embed_query(query)     → query_vector  (1, 384)
  2. index.search(query_vector)      → distances, indices
  3. movie_metadata[idx]             → fetch movie details
  4. return top-k results with rank + distance
```

#### `main.py`

```
Entry point — interactive CLI loop

Startup : initialise MovieSearchEngine (loads + indexes all movies)
Loop    : accept query → call engine.search() → display_results()
Exit    : user types quit / exit / q
```

### Data Structures

**Input row (from CSV)**
```
{
  "Movie Name"  : "Dhurandhar",
  "Description" : "Jaskirat Singh Rangi erases his identity...",
  "Release Date": "2025-01-15",
  "IMDB Rating" : 9.4
}
```

**Search result (returned per movie)**
```
{
  "Movie Name"  : "Dhurandhar",
  "Description" : "Jaskirat Singh Rangi erases his identity...",
  "Release Date": "2025-01-15",
  "IMDB Rating" : 9.4,
  "rank"        : 1,
  "distance"    : 1.1823
}
```

---

## Dependencies

### System Requirements

| Requirement | Minimum Version | Purpose |
|---|---|---|
| **Python** | 3.10 or above | Runtime environment |
| **pip** | Latest | Package installer |
| **RAM** | 4 GB or above | Loading the embedding model into memory |
| **Disk Space** | ~1 GB free | Storing the sentence-transformers model (downloaded once) |
| **Internet Connection** | Required on first run | To download the embedding model from HuggingFace |

> After the first run, the model is cached locally. Internet is not needed for subsequent runs.

### Python Libraries

These are installed automatically via `pip install -r requirements.txt`:

| Library | Version | Purpose |
|---|---|---|
| `pandas` | Latest stable | Read and process the CSV movie dataset |
| `sentence-transformers` | Latest stable | Load the pre-trained embedding model and generate dense vectors |
| `faiss-cpu` | Latest stable | Build the vector index and perform nearest neighbour search |
| `numpy` | Latest stable | Handle float32 array operations for embeddings |

### Embedding Model

| Detail | Value |
|---|---|
| **Model name** | `all-MiniLM-L6-v2` |
| **Source** | HuggingFace (downloaded automatically by sentence-transformers) |
| **Model size** | ~90 MB |
| **Vector dimension** | 384 |
| **Cache location** | `~/.cache/huggingface/hub/` |

> No API key or account is required. The model is open-source and free to use.

### What Gets Downloaded on First Run

```
sentence-transformers
  └── downloads all-MiniLM-L6-v2 from HuggingFace
        ├── config.json
        ├── tokenizer files
        └── model weights (~90 MB)
```

---

## How to Run

### Step 1 — Navigate to the project directory

```bash
cd /path/to/dense_retrieval
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

### Step 4 — Add your movie data

Place your CSV file in the project root. The file must have these columns:

```
Movie Name, Description, Release Date, IMDB Rating
```

> The data loader also supports `Name` as the column header if `Movie Name` is not present.

### Step 5 — Run the program

```bash
python main.py
```

### Step 6 — Search

Once the engine is ready, you will see:

```
============================================================
  Indian Movie Semantic Search (Dense Retrieval)
  Type 'quit' to exit
============================================================

Enter your search query:
```

Type any natural language query and press Enter:

```
Enter your search query: soldier sacrificing life for the country
```

### Sample Output

```
============================================================
#1  Lakshya  (15-Jun-2004)
    IMDB: 8.0  |  Distance: 1.2354
    An aimless young man finds his purpose and becomes a dedicated army officer...

#2  Major  (03-Jun-2022)
    IMDB: 8.2  |  Distance: 1.2394
    Based on the life of Major Sandeep Unnikrishnan, who sacrificed his life...

#3  Dhurandhar  (2025-01-15)
    IMDB: 9.4  |  Distance: 1.2622
    Jaskirat Singh Rangi erases his identity and leaves his family to live as a ghost...
```

### To Exit

```
Enter your search query: quit
```
