"""
dense_retriever.py
------------------
Stage 1-B — Dense Semantic Retrieval (Bi-Encoder)

Concept (§4 of docx, Figure 8-14 of book):
  A bi-encoder encodes the query and each document SEPARATELY into vectors.
  Similarity is computed via cosine distance.

  Why needed alongside BM25:
    BM25 misses synonyms and paraphrases.
    Dense retrieval catches meaning-based matches — same idea, different words.

  Trade-off vs cross-encoder:
    Bi-encoder  → fast (pre-compute doc vectors), approximate meaning match
    Cross-encoder (reranker) → slower, reads query+doc TOGETHER, more accurate
"""
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, texts: list, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[DenseRetriever] Loading bi-encoder '{model_name}' ...")
        self.texts = texts
        self.model = SentenceTransformer(model_name)

        print("[DenseRetriever] Encoding document corpus (pre-computed, stored in index) ...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Normalise once so dot-product == cosine similarity at query time
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.doc_embeddings = embeddings / (norms + 1e-10)
        print(f"[DenseRetriever] Ready — {len(texts)} documents indexed.\n")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        """
        Return top_k candidates by cosine similarity.

        Parameters
        ----------
        query  : natural-language query string
        top_k  : candidates to return for stage-2 reranking

        Returns
        -------
        list of dicts: {index, text, dense_score}
        """
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        scores = self.doc_embeddings @ q_emb          # cosine similarity
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {"index": idx, "text": self.texts[idx], "dense_score": float(score)}
            for idx, score in ranked
        ]
