"""
hybrid_search.py
----------------
Full Search Pipelines

Two pipelines are implemented here:

1. keyword_and_reranking_search  (Figure 8-14 + book code pattern)
   BM25 → top-k candidates → Cross-Encoder Reranker → top-n results
   Matches the book's function of the same name.

2. hybrid_reranking_search  (§4 of docx — Hybrid Search)
   BM25 (keyword match)   ──┐
                             ├──► merge + deduplicate ──► Reranker ──► top-n
   Dense (semantic match) ──┘

Why three stages?
  • The reranker is slow — you cannot run it on all documents.
  • BM25 + Dense act as fast rough filters.
  • The reranker does precise ordering on the small combined pool.
"""


def keyword_and_reranking_search(
    query: str,
    bm25_retriever,
    reranker,
    top_k: int = 10,
    top_n: int = 3,
) -> list:
    """
    Stage 1: BM25 keyword search → top_k candidates
    Stage 2: Cross-encoder reranker → top_n final results

    Direct equivalent of the book's keyword_and_reranking_search function.
    """
    candidates = bm25_retriever.retrieve(query, top_k=top_k)
    return reranker.rerank(query, candidates, top_n=top_n)


def hybrid_reranking_search(
    query: str,
    bm25_retriever,
    dense_retriever,
    reranker,
    top_k: int = 10,
    top_n: int = 3,
) -> list:
    """
    Stage 1A: BM25 retrieval        → top_k keyword candidates
    Stage 1B: Dense retrieval       → top_k semantic candidates
    Stage 2:  Merge + deduplicate   → ~2×top_k unique candidates
    Stage 3:  Cross-encoder reranker→ top_n final results

    BM25 alone misses synonyms/paraphrases.
    Dense alone misses exact rare terms.
    Together they cover both gaps before the reranker does precise ordering.
    """
    # Stage 1: two retrievers run on the same query
    bm25_results = bm25_retriever.retrieve(query, top_k=top_k)
    dense_results = dense_retriever.retrieve(query, top_k=top_k)

    # Stage 2: merge, keep first occurrence (preserves best individual score)
    seen = set()
    candidates = []
    for result in bm25_results + dense_results:
        if result["index"] not in seen:
            seen.add(result["index"])
            candidates.append(result)

    # Stage 3: reranker decides the final order
    return reranker.rerank(query, candidates, top_n=top_n)
