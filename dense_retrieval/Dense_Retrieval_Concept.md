# Dense Retrieval — Concepts

---

## 1. What is Dense Retrieval?

Dense retrieval is a technique to search for relevant information by comparing the **meaning** of a query against a collection of documents — rather than matching exact words.

The word **dense** refers to the vector representation used. Each piece of text is converted into a **dense vector** — an array of hundreds of floating point numbers where every dimension carries meaning.

```
"A soldier sacrifices his life for the country"
                    ↓
        [0.23, -0.11, 0.87, 0.04, ...]   ← 384 dense numbers
```

This is in contrast to **sparse** representations (like TF-IDF or BM25) where most values in the vector are zero.

---

## 2. Embeddings — The Core Idea

An **embedding** is a numeric representation of text in a continuous vector space. It is produced by passing text through a pre-trained language model.

### Key Properties

- Two pieces of text with **similar meaning** will have vectors that are **close together** in vector space.
- Two pieces of text with **different meaning** will have vectors that are **far apart**.

### Visual Intuition

```
         Vector Space

    Text 1 ●
    Text 2   ●          ← Text 1 and Text 2 are close → similar meaning

                  ● Text 3   ← Text 3 is far → different meaning
```

### Example

| Text | Meaning |
|---|---|
| "A soldier dies for his country" | Patriotic sacrifice |
| "An army officer gives his life for the nation" | Patriotic sacrifice |
| "A chef cooks pasta in Italy" | Unrelated |

Text 1 and Text 2 will have very similar vectors even though they share no common words. Text 3 will be far away from both.

---

## 3. Keyword Search vs Dense Retrieval

| | Keyword Search | Dense Retrieval |
|---|---|---|
| **Matching** | Exact word match | Semantic / meaning match |
| **Representation** | Sparse vector (mostly zeros) | Dense vector (all values filled) |
| **Index** | Inverted index | Vector index (FAISS, Weaviate) |
| **Query: "soldier sacrifice"** | Only finds docs with those exact words | Finds docs about army, mission, patriotism |
| **Handles synonyms** | No | Yes |
| **Handles paraphrasing** | No | Yes |
| **Speed at scale** | Fast | Fast with ANN (Approximate Nearest Neighbour) |

### When Keyword Search Fails

Query: *"how precise was the science"*

- Keyword search returns documents containing "precise" and "science"
- Dense retrieval understands the question and returns the most contextually relevant passage

---

## 4. Chunking

Language models have a **maximum input length** (context window). Long documents must be broken into smaller pieces called **chunks** before embedding.

### Why Chunking Matters

- Embedding a very long document into a single vector causes **information loss** — the model compresses too much into one vector
- Smaller, focused chunks produce more **precise embeddings**
- At search time, the system can pinpoint the exact chunk that answers the query

### Chunking Strategies

#### One Vector per Document
- Embed the entire document as a single vector
- Simple but loses detail for long documents
- Good for short texts like movie descriptions

```
Document → [single vector]
```

#### Multiple Vectors per Document (Chunking)
- Split document into chunks, embed each chunk separately
- More granular, better for long documents
- Each chunk gets its own vector in the index

```
Document → Chunk 1 → [vector 1]
         → Chunk 2 → [vector 2]
         → Chunk 3 → [vector 3]
```

#### Chunking Approaches

| Approach | How it Works | Best For |
|---|---|---|
| **Sentence split** | Each sentence is a chunk | Precise retrieval |
| **Token split** | Fixed number of tokens per chunk | Uniform chunk sizes |
| **Character split** | Fixed number of characters per chunk | Simple implementation |
| **Overlapping window** | Chunks overlap by N tokens | Preserving context at boundaries |

### Overlapping Chunks — Why It Helps

Without overlap, context at chunk boundaries is lost:

```
Chunk 1: "Llama 2 was trained on"
Chunk 2: "40% more data than Llama"
```

With overlap, boundary context is preserved:

```
Chunk 1: "Llama 2 was trained on"
Chunk 2: "trained on 40% more data than Llama"
```

---

## 5. Nearest Neighbour Search

Once all chunks are embedded, we need to find the chunk most similar to a query. This is the **nearest neighbour search** problem.

### L2 Distance (Euclidean)

The most common similarity measure. Lower distance = more similar.

```
distance = sqrt( (a1-b1)² + (a2-b2)² + ... + (an-bn)² )
```

### Cosine Similarity

Measures the angle between two vectors. Higher score = more similar. Often used when vector magnitude should be ignored.

### NumPy (Brute Force)

For small datasets, calculate distances against every vector:

```python
distances = np.linalg.norm(embeddings - query_embedding, axis=1)
```

Simple but slow for large datasets — checks every single vector.

---

## 6. Vector Databases

For large-scale datasets (millions of vectors), brute force search is too slow. **Vector databases** solve this with optimised indexing structures.

### FAISS (Facebook AI Similarity Search)

Used in this project. An open-source library for fast similarity search.

```python
index = faiss.IndexFlatL2(dim)   # exact L2 search
index.add(embeddings)            # store vectors
index.search(query, top_k)       # find nearest neighbours
```

- Stores the embeddings in an optimised structure
- Can retrieve nearest neighbours in milliseconds even for very large collections
- Supports both exact and approximate search

### Other Vector Databases

| Database | Type | Best For |
|---|---|---|
| **FAISS** | Library | Local, research, prototyping |
| **Weaviate** | Cloud / Self-hosted | Production, filtering, metadata |
| **Pinecone** | Cloud | Managed, scalable production |
| **Chroma** | Library | Local, simple applications |

### Advantages Over NumPy

| | NumPy | Vector Database |
|---|---|---|
| **Scale** | Thousands of vectors | Millions of vectors |
| **Speed** | Slow (brute force) | Fast (optimised index) |
| **Features** | None | Filter, delete, update vectors |
| **Production ready** | No | Yes |

---

## 7. Fine-Tuning Embedding Models for Dense Retrieval

Pre-trained embedding models are general purpose. Fine-tuning improves them for a **specific domain or task**.

### The Problem with General Models

A general model trained on web text may not understand domain-specific language — legal terms, medical jargon, or niche topics.

### How Fine-Tuning Works

Fine-tuning uses **triplets** of training examples:

```
Anchor (document)   : "Interstellar premiered on October 26, 2014, in Los Angeles"
Positive (relevant) : "When did Interstellar release date"      ← relevant query
Negative (irrelevant): "Interstellar cast"                      ← irrelevant query
```

The model is trained to:
- Pull the **anchor** and **positive** closer together in vector space
- Push the **anchor** and **negative** further apart

### Before vs After Fine-Tuning

**Before fine-tuning:**
```
Vector Space:

  [Document] ●
                ● [Relevant Query]     ← too far
                ● [Irrelevant Query]   ← too close
```

**After fine-tuning:**
```
Vector Space:

  [Document] ●
             ● [Relevant Query]        ← pulled closer
                        ● [Irrelevant Query]   ← pushed further
```

### Result

The fine-tuned model produces embeddings where relevant queries are **closer** to the document and irrelevant queries are **further away** — improving retrieval accuracy on that domain.

---

## 8. Reranking

Dense retrieval returns the top-k most similar documents. But similarity in vector space does not always mean the best answer. **Reranking** is a post-retrieval step that re-scores the results more carefully.

### Why Reranking?

- The embedding model compresses text into a fixed-size vector — some nuance is lost
- A **cross-encoder** (reranker) looks at the query and each result **together**, producing a more accurate relevance score
- It is slower than retrieval but only runs on the small top-k set

### Two-Stage Pipeline

```
Stage 1 — Dense Retrieval (fast, approximate)
  Query → embed → FAISS search → Top-100 candidates

Stage 2 — Reranking (slower, precise)
  (Query + each candidate) → cross-encoder → relevance score → Top-5 results
```

### Retrieval vs Reranking

| | Dense Retrieval | Reranking |
|---|---|---|
| **Model type** | Bi-encoder (query and doc embedded separately) | Cross-encoder (query + doc together) |
| **Speed** | Fast | Slower |
| **Accuracy** | Good | Better |
| **Runs on** | Entire corpus | Only top-k candidates |

---

## 9. How All Concepts Connect

```
Raw Documents (CSV)
        │
        ▼
   ┌─────────┐
   │ Chunking│  ← split long docs into meaningful pieces
   └────┬────┘
        │
        ▼
   ┌──────────────────┐
   │ Embedding Model  │  ← convert each chunk to dense vector
   │ (fine-tuned for  │
   │  better results) │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Vector Database │  ← store and index all vectors (FAISS)
   │  (FAISS / etc.)  │
   └────────┬─────────┘
            │
            ◀─── User Query (also embedded by same model)
            │
            ▼
   ┌──────────────────┐
   │ Nearest Neighbour│  ← find top-k closest vectors
   │     Search       │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │    Reranking     │  ← optional: re-score top-k for better precision
   └────────┬─────────┘
            │
            ▼
      Final Results
```

---

## 10. Summary

| Concept | What It Does | Used In This Project |
|---|---|---|
| **Embeddings** | Convert text to dense vectors | `embedder.py` |
| **Chunking** | Split documents into searchable pieces | `data_loader.py` |
| **Nearest Neighbour Search** | Find most similar vectors | `search_index.py` |
| **Vector Database (FAISS)** | Store and index vectors efficiently | `search_index.py` |
| **Fine-Tuning** | Improve embeddings for specific domain | Concept reference |
| **Reranking** | Re-score top-k results for better accuracy | Concept reference |
