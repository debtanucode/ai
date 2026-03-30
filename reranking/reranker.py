"""
reranker.py
-----------
Stage 2 — Cross-Encoder Reranker

Concept (§3 of docx, Figure 8-15 of book):
  A cross-encoder feeds the QUERY + DOCUMENT together as one combined input
  into the model. Full attention across both texts → more accurate than
  a bi-encoder, but slower (cannot pre-compute; must run per pair).

  This is exactly what Cohere's Rerank endpoint does internally.
  Open-source equivalent: sentence-transformers CrossEncoder.

  Reranking pipeline (book's keyword_and_reranking_search pattern):
    results = co.rerank(query=query, documents=docs, top_n=3, return_documents=True)
    ↕  (same idea, different library)
    results = reranker.rerank(query, candidates, top_n=3)

  Key boundary (§2 of docx):
    The reranker is NOT a retriever. It can only REORDER what is passed to it.
    It cannot fetch new documents.

Relevance score (§3 of docx):
  Output is a float 0–1 (via sigmoid). Closer to 1 = highly relevant.
  RELATIVE ordering matters more than the absolute value.
"""
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Reranker] Loading cross-encoder '{model_name}' ...")
        self.model = CrossEncoder(model_name)
        print("[Reranker] Ready.\n")

    def rerank(self, query: str, candidates: list, top_n: int = 3) -> list:
        """
        Score every candidate against the query and return top_n sorted by
        relevance score descending.

        Mirrors the book's API:
          results = co.rerank(query, documents, top_n, return_documents=True)

        Parameters
        ----------
        query      : user query
        candidates : list of dicts, each must have 'index' and 'text'
        top_n      : how many final results to return (narrow down hard)

        Returns
        -------
        list of dicts: candidate dict + 'relevance_score' (float 0–1)
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        raw_scores = self.model.predict(pairs)                 # logits
        relevance_scores = 1 / (1 + np.exp(-raw_scores))      # sigmoid → 0–1

        results = []
        for i, candidate in enumerate(candidates):
            results.append({**candidate, "relevance_score": float(relevance_scores[i])})

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_n]
