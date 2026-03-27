"""
03 — Semantic Movie Search
===========================
Covers:
  - Text embeddings: one vector for an entire sentence
  - How mean-pooling collapses [N tokens × 768 dims] → [768 dims]
  - Cosine similarity to rank documents by meaning
  - Why semantic search finds matches keyword search misses

Dataset : indian_movies.csv  (20 Indian movies)
Model   : sentence-transformers/all-MiniLM-L6-v2  (384-dim, ~90 MB)

Usage:
  python src/03_semantic_search.py
  Then type any query and press Enter.  Type 'quit' to exit.
"""

import csv
import os
import numpy as np
from sentence_transformers import SentenceTransformer

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "indian_movies.csv")


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_movies(path: str) -> list[dict]:
    movies = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movies.append(row)
    return movies


# ── Cosine similarity ─────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Search ────────────────────────────────────────────────────────────────────
def search(query: str, movies: list[dict], embeddings: np.ndarray,
           model: SentenceTransformer, top_k: int = 3) -> list[dict]:
    query_vec = model.encode(query, normalize_embeddings=True)
    scores = [
        cosine_similarity(query_vec, embeddings[i])
        for i in range(len(movies))
    ]
    ranked = sorted(range(len(movies)), key=lambda i: scores[i], reverse=True)
    return [(movies[i], scores[i]) for i in ranked[:top_k]]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    movies = load_movies(CSV_PATH)
    print(f"Loaded {len(movies)} movies from CSV.")

    # Encode all descriptions into text embeddings
    descriptions = [m["Description"] for m in movies]
    print("Encoding movie descriptions into text embeddings...")
    embeddings = model.encode(descriptions, normalize_embeddings=True)

    print(f"\n── Text Embedding Shape ─────────────────────────────────────────────")
    print(f"  Each description → 1 vector of {embeddings.shape[1]} dims")
    print(f"  All movies matrix : {embeddings.shape}  ← [20 movies × 384 dims]")
    print(f"\n  How it's made (mean-pooling):")
    print(f"  'Best' → [0.23, 0.81, ...]")
    print(f"  'movie'→ [0.54, 0.12, ...]")
    print(f"  'ever' → [0.11, 0.94, ...]")
    print(f"  ─────────────────────────")
    print(f"  Average→ [0.29, 0.62, ...]  ← one vector = text embedding")

    print("\n" + "═" * 60)
    print("  SEMANTIC MOVIE SEARCH")
    print("  Find movies by meaning, not keywords.")
    print("═" * 60)
    print("  Try queries like:")
    print("    'spy who sacrifices everything for the nation'")
    print("    'soldier fighting in a war'")
    print("    'man fights against terrorists alone'")
    print("    'nuclear weapons and secret mission'")
    print("  Type 'quit' to exit.\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = search(query, movies, embeddings, model, top_k=3)

        print(f"\n  Top 3 matches for: '{query}'")
        print(f"  {'#':<3} {'Score':<8} {'IMDB':<6} {'Movie'}")
        print(f"  {'─'*3} {'─'*7} {'─'*5} {'─'*35}")
        for rank, (movie, score) in enumerate(results, 1):
            print(f"  {rank:<3} {score:.4f}   {movie['IMDB Rating']:<6} {movie['Movie Name']}")
            # Show a snippet of the description
            desc = movie["Description"]
            snippet = desc[:90] + "..." if len(desc) > 90 else desc
            print(f"      ↳ {snippet}")
        print()


if __name__ == "__main__":
    main()
