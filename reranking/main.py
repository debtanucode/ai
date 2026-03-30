"""
main.py  —  Semantic Search + Reranking on Indian Movies
=========================================================
Single file that runs every feature implemented in this project.

Features covered
----------------
  1.  Data Loading           — load indian_movies.csv
  2.  BM25 Retrieval         — keyword-based first-stage retrieval
  3.  Dense Retrieval        — embedding-based first-stage retrieval (bi-encoder)
  4.  Cross-Encoder Reranker — reads query + doc together, outputs relevance score 0-1
  5.  keyword_and_reranking_search — BM25 → reranker  (book pattern, Figure 8-14)
  6.  hybrid_reranking_search      — BM25 + Dense → merge → reranker (full pipeline)
  7.  Precision@k            — fraction of top-k results that are relevant
  8.  Average Precision (AP) — precision averaged at relevant positions (rewards early hits)
  9.  MAP                    — Mean AP across all test-suite queries (system benchmark)
  10. Prompt-Based LLM Reranker — LLM scores relevance via a natural-language prompt

Run
---
  pip install rank-bm25 sentence-transformers numpy anthropic
  python main.py

  For Feature 10 set your Anthropic API key:
    export ANTHROPIC_API_KEY="sk-ant-..."
  Without the key Feature 10 shows the concept and prompt template only.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_FILE       = Path(__file__).parent / "indian_movies.csv"
TOP_K           = 10    # first-stage candidates (cast a wide net)
TOP_N           = 3     # final results after reranking (narrow down hard)
DENSE_MODEL     = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SEP  = "═" * 62
SEP2 = "─" * 58


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 1 — DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_movies():
    """Load movies from CSV. Returns (movies list, descriptions list)."""
    movies = []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            movies.append(dict(row))
    texts = [m["Description"] for m in movies]
    return movies, texts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 2 — BM25 RETRIEVAL  (Stage 1-A)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concept: counts how often query words appear in each document.
# Fast, but matches WORDS not MEANING — misses synonyms and paraphrases.

class BM25Retriever:
    def __init__(self, texts):
        self.texts = texts
        self.bm25  = BM25Okapi([t.lower().split() for t in texts])

    def retrieve(self, query, top_k=10):
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"index": i, "text": self.texts[i], "bm25_score": float(s)} for i, s in ranked]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 3 — DENSE RETRIEVAL  (Stage 1-B, Bi-Encoder)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concept: encodes query and documents SEPARATELY into vectors.
# Catches meaning-based matches BM25 misses ("spy" ≈ "undercover agent").
# Trade-off: faster than cross-encoder, less accurate (bi-encoder).

class DenseRetriever:
    def __init__(self, texts, model_name=DENSE_MODEL):
        print(f"  [DenseRetriever] Loading bi-encoder '{model_name}' ...")
        self.texts = texts
        self.model = SentenceTransformer(model_name)
        print("  [DenseRetriever] Encoding corpus (stored in index, not re-run per query) ...")
        emb   = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.doc_emb = emb / (norms + 1e-10)   # normalised → dot-product == cosine
        print(f"  [DenseRetriever] Ready — {len(texts)} documents indexed.\n")

    def retrieve(self, query, top_k=10):
        q = self.model.encode([query], convert_to_numpy=True)[0]
        q = q / (np.linalg.norm(q) + 1e-10)
        scores = self.doc_emb @ q
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"index": i, "text": self.texts[i], "dense_score": float(s)} for i, s in ranked]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 4 — CROSS-ENCODER RERANKER  (Stage 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concept: feeds QUERY + DOCUMENT together as one input (not separately).
# Full attention across both texts → more accurate than bi-encoder.
# Output: relevance score 0–1 per document (sigmoid of raw logit).
# Important boundary: NOT a retriever — can only reorder what is passed to it.
# Open-source equivalent of Cohere's co.rerank() endpoint.

class CrossEncoderReranker:
    def __init__(self, model_name=RERANKER_MODEL):
        print(f"  [Reranker] Loading cross-encoder '{model_name}' ...")
        self.model = CrossEncoder(model_name)
        print("  [Reranker] Ready.\n")

    def rerank(self, query, candidates, top_n=3):
        if not candidates:
            return []
        raw    = self.model.predict([(query, c["text"]) for c in candidates])
        scores = 1 / (1 + np.exp(-raw))   # sigmoid → 0–1
        results = [{**c, "relevance_score": float(scores[i])} for i, c in enumerate(candidates)]
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 10 — PROMPT-BASED LLM RERANKER  (§3 of docx)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Concept: send query + document to an LLM with a natural-language prompt that
# asks it to rate relevance 0-1. Same idea as cross-encoder, but:
#   • Pro  — easy to customise the scoring criteria in plain English
#   • Con  — one full LLM call per document → much slower and more expensive
#
# Prompt template (from docx §3):
#   Given this query: "{query}"
#   And this document: "{document}"
#   Rate how relevant the document is to the query.
#   Return ONLY a JSON object like {"score": 0.85}
#
# Cross-encoder vs Prompt-based LLM:
#   Cross-encoder  → small dedicated model, fast, cheap, less customisable
#   Prompt LLM     → large general model, slow, expensive, fully customisable

_LLM_PROMPT_TEMPLATE = (
    'Given this query: "{query}"\n'
    'And this document: "{document}"\n'
    "Rate how relevant the document is to the query.\n"
    'Return ONLY a JSON object like {{"score": 0.85}} with a value between 0 and 1.'
)

LLM_RERANKER_MODEL = "claude-haiku-4-5-20251001"   # fastest/cheapest Claude model


class LLMReranker:
    """
    Reranks candidates by asking an LLM to score each (query, document) pair.

    Requires:
      • anthropic package  (`pip install anthropic`)
      • ANTHROPIC_API_KEY  environment variable set
    """

    def __init__(self, model: str = LLM_RERANKER_MODEL):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model

    def _score(self, query: str, document: str) -> float:
        """Ask the LLM to rate relevance of one (query, document) pair → float 0-1."""
        prompt = _LLM_PROMPT_TEMPLATE.format(query=query, document=document)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        try:
            return float(json.loads(raw)["score"])
        except Exception:
            # Fallback: try to parse a bare float from the response
            try:
                return float(raw)
            except Exception:
                return 0.0

    def rerank(self, query: str, candidates: list, top_n: int = 3) -> list:
        """Score every candidate with the LLM, return top_n sorted by llm_score."""
        scored = []
        for c in candidates:
            score = self._score(query, c["text"])
            scored.append({**c, "llm_score": score})
        scored.sort(key=lambda x: x["llm_score"], reverse=True)
        return scored[:top_n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 5 — keyword_and_reranking_search  (Figure 8-14 book pattern)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 1: BM25 → top_k candidates
# Stage 2: Cross-encoder reranker → top_n final results

def keyword_and_reranking_search(query, bm25, reranker, top_k=TOP_K, top_n=TOP_N):
    candidates = bm25.retrieve(query, top_k=top_k)
    return reranker.rerank(query, candidates, top_n=top_n)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 6 — hybrid_reranking_search  (Full 3-stage pipeline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stage 1A: BM25     (keyword match)  → top_k
# Stage 1B: Dense    (semantic match) → top_k
# Stage 2:  Merge + deduplicate       → ~2×top_k unique candidates
# Stage 3:  Cross-encoder reranker    → top_n final results

def hybrid_reranking_search(query, bm25, dense, reranker, top_k=TOP_K, top_n=TOP_N):
    bm25_hits  = bm25.retrieve(query, top_k=top_k)
    dense_hits = dense.retrieve(query, top_k=top_k)
    seen, pool = set(), []
    for r in bm25_hits + dense_hits:
        if r["index"] not in seen:
            seen.add(r["index"])
            pool.append(r)
    return reranker.rerank(query, pool, top_n=top_n)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 7 — Precision@k
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Of the top-k results, what fraction are actually relevant?
# Precision@k = relevant hits in top-k / k
# Weakness: position-blind — a hit at #1 and at #3 score the same.

def precision_at_k(retrieved, relevant, k):
    return sum(1 for idx in retrieved[:k] if idx in relevant) / k if k else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 8 — Average Precision (AP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Precision recorded only at positions where a relevant doc appears, then averaged.
# Rewards systems that rank relevant docs HIGHER.
# System A [✓,✗,✓] → P@1=1.0, P@3=0.67 → AP=(1.0+0.67)/total_relevant
# System B [✗,✗,✓] → P@3=0.33          → AP=0.33/total_relevant

def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    hits, precisions = 0, []
    for rank, idx in enumerate(retrieved, 1):
        if idx in relevant:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / len(relevant)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE 9 — Mean Average Precision (MAP)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AP extended across ALL queries in the test suite — one number for the system.
# MAP = (AP_q1 + AP_q2 + ... + AP_qn) / n

def mean_average_precision(pairs):
    aps = [average_precision(r, rel) for r, rel in pairs]
    return sum(aps) / len(aps) if aps else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST SUITE  (Figure 8-16 — answer key, query → relevant movie indices)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST_SUITE = [
    {
        "query":    "covert spy mission in Pakistan",
        "relevant": {0, 1, 2, 8, 10},   # Dhurandhar, Dhurandhar:Revenge, 16 Dec, Article 370, Madras Cafe
    },
    {
        "query":    "soldiers battle war sacrifice",
        "relevant": {12, 13, 14, 15, 16, 17, 18, 19},  # Uri, Shershaah, Border, Sam Bahadur, 1971, Major, Ghazi, Lakshya
    },
    {
        "query":    "freedom fighter revolutionary assassination",
        "relevant": {3, 7, 11},          # Sardar Udham, Sarfarosh, Bhagat Singh
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS — pretty printing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_results(results, movies, label):
    print(f"\n  {SEP2}")
    print(f"  {label}")
    print(f"  {SEP2}")
    for i, r in enumerate(results, 1):
        movie = movies[r["index"]]
        key   = "relevance_score" if "relevance_score" in r else \
                "llm_score"       if "llm_score"       in r else \
                "dense_score"     if "dense_score"     in r else "bm25_score"
        print(f"  #{i}  [{r[key]:.4f}]  {movie['Movie Name']}")
        print(f"       {r['text'][:72]}...")

def print_ap_walkthrough(retrieved, relevant, label):
    print(f"\n  Walkthrough — {label}")
    print(f"  {SEP2}")
    print(f"  Retrieved : {retrieved}")
    print(f"  Relevant  : {sorted(relevant)}")
    hits, ps = 0, []
    for rank, idx in enumerate(retrieved, 1):
        mark = "✓" if idx in relevant else "✗"
        if idx in relevant:
            hits += 1
            p = hits / rank
            ps.append(p)
            print(f"    Rank {rank}: idx={idx} {mark}  →  P@{rank} = {hits}/{rank} = {p:.2f}  (recorded)")
        else:
            print(f"    Rank {rank}: idx={idx} {mark}  →  skipped")
    ap = sum(ps) / len(relevant) if relevant else 0.0
    expr = " + ".join(f"{p:.2f}" for p in ps) if ps else "0"
    print(f"  AP = ({expr}) / {len(relevant)} = {ap:.3f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN — runs every feature in sequence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():

    # ── FEATURE 1: Load Data ─────────────────────────────────────────────────
    print_header("FEATURE 1 — Data Loading")
    movies, texts = load_movies()
    print(f"\n  File    : {DATA_FILE.name}")
    print(f"  Records : {len(movies)}")
    print(f"  Columns : {', '.join(movies[0].keys())}")
    print(f"\n  Sample documents (used as search corpus):")
    for i, m in enumerate(movies[:3]):
        print(f"    [{i}] {m['Movie Name']} — {m['Description'][:60]}...")
    print(f"    ...")

    # ── GET USER QUERY ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  ENTER YOUR SEARCH QUERY")
    print(f"{SEP}")
    print(f"\n  Tip — the test suite has 3 queries with known relevant documents.")
    print(f"  Enter one of them to also see P@k / AP evaluated on your results:")
    for s in TEST_SUITE:
        print(f"    • {s['query']}")
    print()
    user_query = input("  Your query: ").strip()
    if not user_query:
        user_query = "covert spy mission in Pakistan"
        print(f"  (No input — using default: \"{user_query}\")")
    print()

    # ── FEATURE 2: BM25 Retrieval ────────────────────────────────────────────
    print_header("FEATURE 2 — BM25 Retrieval  (keyword-based, Stage 1-A)")
    bm25 = BM25Retriever(texts)
    bm25_only = bm25.retrieve(user_query, top_k=TOP_N)
    print(f'\n  Query: "{user_query}"')
    print(f"  (Scores documents by word overlap — fast but meaning-blind)")
    print_results(bm25_only, movies, f"BM25 top-{TOP_N}")

    # ── FEATURE 3: Dense Retrieval ───────────────────────────────────────────
    print_header("FEATURE 3 — Dense Retrieval  (bi-encoder, Stage 1-B)")
    print()
    dense = DenseRetriever(texts)
    dense_only = dense.retrieve(user_query, top_k=TOP_N)
    print(f'  Query: "{user_query}"')
    print(f"  (Encodes meaning — catches matches BM25 misses)")
    print_results(dense_only, movies, f"Dense top-{TOP_N}")

    # ── FEATURE 4: Cross-Encoder Reranker ───────────────────────────────────
    print_header("FEATURE 4 — Cross-Encoder Reranker  (Stage 2)")
    print()
    reranker = CrossEncoderReranker()
    all_candidates = bm25.retrieve(user_query, top_k=TOP_K)
    reranked = reranker.rerank(user_query, all_candidates, top_n=TOP_N)
    print(f'  Query: "{user_query}"')
    print(f"  Input : {TOP_K} BM25 candidates  →  reranker reads (query + each doc) together")
    print(f"  Output: top-{TOP_N} sorted by relevance score 0–1")
    print_results(reranked, movies, f"Cross-Encoder top-{TOP_N}  (from {TOP_K} BM25 candidates)")

    # ── FEATURE 5: keyword_and_reranking_search ──────────────────────────────
    print_header("FEATURE 5 — keyword_and_reranking_search  (Figure 8-14 pattern)")
    kw_rerank = keyword_and_reranking_search(user_query, bm25, reranker)
    print(f'\n  Query: "{user_query}"')
    print(f"  Pipeline: BM25 (top-{TOP_K}) → Cross-Encoder Reranker → top-{TOP_N}")
    print_results(kw_rerank, movies, "BM25 → Reranker")

    # ── FEATURE 6: hybrid_reranking_search ──────────────────────────────────
    print_header("FEATURE 6 — hybrid_reranking_search  (Full 3-stage pipeline)")
    hybrid = hybrid_reranking_search(user_query, bm25, dense, reranker)
    bm25_c  = bm25.retrieve(user_query, top_k=TOP_K)
    dense_c = dense.retrieve(user_query, top_k=TOP_K)
    pool_size = len({r["index"] for r in bm25_c + dense_c})
    print(f'\n  Query: "{user_query}"')
    print(f"  Stage 1A : BM25   → {TOP_K} candidates")
    print(f"  Stage 1B : Dense  → {TOP_K} candidates")
    print(f"  Stage 2  : Merge + deduplicate → {pool_size} unique candidates")
    print(f"  Stage 3  : Reranker → top-{TOP_N} final results")
    print_results(hybrid, movies, "Hybrid (BM25 + Dense) → Reranker")

    # ── Resolve eval query for Features 7 & 8 ────────────────────────────────
    # Precision@k and AP require a known relevant-document set (answer key).
    # If the user's query matches a TEST_SUITE entry we use those results directly.
    # Otherwise we fall back to TEST_SUITE[0] and re-run search so the demo is
    # still meaningful.
    suite_match = next(
        (s for s in TEST_SUITE if s["query"].lower() == user_query.lower()), None
    )
    if suite_match:
        eval_query    = user_query
        demo_relevant = suite_match["relevant"]
        eval_bm25     = bm25_only
        eval_kw       = kw_rerank
        eval_hybrid   = hybrid
    else:
        eval_query    = TEST_SUITE[0]["query"]
        demo_relevant = TEST_SUITE[0]["relevant"]
        eval_bm25     = bm25.retrieve(eval_query, top_k=TOP_N)
        eval_kw       = keyword_and_reranking_search(eval_query, bm25, reranker)
        eval_hybrid   = hybrid_reranking_search(eval_query, bm25, dense, reranker)

    # ── FEATURE 7: Precision@k ───────────────────────────────────────────────
    print_header("FEATURE 7 — Precision@k  (position-blind metric)")
    if not suite_match:
        print(f'\n  Note: "{user_query[:55]}" is not in the test suite')
        print(f'  (no pre-defined answer key). Showing evaluation with:')
        print(f'  "{eval_query}"')
    print(f'\n  Query   : "{eval_query}"')
    print(f"  Relevant: {sorted(demo_relevant)}")
    for label, results in [
        ("BM25 Only",         eval_bm25),
        ("BM25 + Reranker",   eval_kw),
        ("Hybrid + Reranker", eval_hybrid),
    ]:
        idx = [r["index"] for r in results]
        p = precision_at_k(idx, demo_relevant, k=TOP_N)
        hits = [i for i in idx if i in demo_relevant]
        print(f"\n  {label}")
        print(f"    Retrieved : {idx}")
        print(f"    Hits      : {hits}")
        print(f"    Precision@{TOP_N} = {len(hits)}/{TOP_N} = {p:.2f}")
    print(f"\n  Weakness: does not reward position — a hit at #1 == hit at #{TOP_N}.")
    print(f"  → Average Precision (next) fixes this.")

    # ── FEATURE 8: Average Precision ────────────────────────────────────────
    print_header("FEATURE 8 — Average Precision  (rewards early relevant hits)")
    if not suite_match:
        print(f'\n  (Using test-suite query for evaluation — see note in Feature 7)')
    print(f'\n  Query   : "{eval_query}"   |   Relevant: {sorted(demo_relevant)}')
    for label, results in [
        ("BM25 Only",         eval_bm25),
        ("BM25 + Reranker",   eval_kw),
        ("Hybrid + Reranker", eval_hybrid),
    ]:
        print_ap_walkthrough([r["index"] for r in results], demo_relevant, label)

    # ── FEATURE 9: MAP across full test suite ────────────────────────────────
    print_header("FEATURE 9 — MAP  (Mean Average Precision, Figure 8-23)")
    print("\n  Test Suite (3 queries with known relevant documents):")
    for item in TEST_SUITE:
        print(f"    Query   : \"{item['query']}\"")
        print(f"    Relevant: {sorted(item['relevant'])}\n")

    def bm25_fn(q):   return bm25.retrieve(q, top_k=TOP_N)
    def kw_fn(q):     return keyword_and_reranking_search(q, bm25, reranker)
    def hybrid_fn(q): return hybrid_reranking_search(q, bm25, dense, reranker)

    map_scores = {}
    for name, fn in [
        ("BM25 Only",         bm25_fn),
        ("BM25 + Reranker",   kw_fn),
        ("Hybrid + Reranker", hybrid_fn),
    ]:
        pairs = []
        print(f"  ── {name} ──")
        for item in TEST_SUITE:
            res = fn(item["query"])
            idx = [r["index"] for r in res]
            ap  = average_precision(idx, item["relevant"])
            pk  = precision_at_k(idx, item["relevant"], k=TOP_N)
            pairs.append((idx, item["relevant"]))
            print(f"    \"{item['query'][:40]}...\"  P@{TOP_N}={pk:.2f}  AP={ap:.3f}")
        m = mean_average_precision(pairs)
        map_scores[name] = m
        print(f"    MAP = {m:.3f}\n")

    # ── FEATURE 10: Prompt-Based LLM Reranker ────────────────────────────────
    print_header("FEATURE 10 — Prompt-Based LLM Reranker  (§3 of docx)")
    print(f'\n  Query: "{user_query}"')
    print(f"\n  How it works:")
    print(f"    For each (query, document) pair the LLM is asked:")
    print(f"    ┌─────────────────────────────────────────────────────────┐")
    print(f'    │ Given this query: "{{query}}"                            │')
    print(f'    │ And this document: "{{document}}"                        │')
    print(f"    │ Rate how relevant the document is to the query.         │")
    print(f'    │ Return ONLY a JSON object like {{"score": 0.85}}          │')
    print(f"    └─────────────────────────────────────────────────────────┘")
    print(f"\n  Cross-encoder vs Prompt-based LLM reranker:")
    print(f"  {'Approach':<26}  {'Speed':<10}  {'Cost':<10}  {'Customisable'}")
    print(f"  {'─'*26}  {'─'*10}  {'─'*10}  {'─'*12}")
    print(f"  {'Cross-encoder model':<26}  {'Fast':<10}  {'Low':<10}  No  (fixed model)")
    print(f"  {'Prompt-based LLM':<26}  {'Slow':<10}  {'High':<10}  Yes (edit the prompt)")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not _ANTHROPIC_AVAILABLE:
        print(f"\n  [SKIPPED] anthropic package not installed.")
        print(f"  Run: pip install anthropic   then set ANTHROPIC_API_KEY")
    elif not api_key:
        print(f"\n  [SKIPPED] ANTHROPIC_API_KEY not set.")
        print(f"  Set it with: export ANTHROPIC_API_KEY=\"sk-ant-...\"")
        print(f"  The code and class are ready — Feature 10 will run once the key is set.")
    else:
        print(f"\n  Model : {LLM_RERANKER_MODEL}")
        print(f"  Scoring {TOP_K} BM25 candidates — one LLM call per document ...")
        llm_reranker = LLMReranker()
        llm_candidates = bm25.retrieve(user_query, top_k=TOP_K)
        llm_results = llm_reranker.rerank(user_query, llm_candidates, top_n=TOP_N)
        print_results(llm_results, movies, f"LLM Reranker top-{TOP_N}  (from {TOP_K} BM25 candidates)")

    # ── Final comparison ─────────────────────────────────────────────────────
    print_header("SUMMARY — MAP Comparison across all 3 pipelines")
    print()
    for name, score in map_scores.items():
        bar = "█" * int(score * 35)
        print(f"  {name:<22}  MAP={score:.3f}  {bar}")

    best = max(map_scores, key=map_scores.get)
    print(f"\n  Best system: {best}")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
