"""
bm25_retriever.py
-----------------
Stage 1-A — BM25 Keyword Retrieval

Concept (§1 of the docx):
  BM25 is a keyword-based retrieval algorithm. It scores documents by counting
  how often query words appear in them. It is fast but matches WORDS, not meaning.

  Limitation: "spy undercover mission" will miss documents that say
  "covert agent operation" — zero word overlap, zero BM25 score.
"""
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, texts: list):
        self.texts = texts
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list:
        """
        Return top_k candidate documents ranked by BM25 score.

        Parameters
        ----------
        query  : natural-language query string
        top_k  : how many candidates to return (cast a wide net at stage 1)

        Returns
        -------
        list of dicts: {index, text, bm25_score}
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {"index": idx, "text": self.texts[idx], "bm25_score": float(score)}
            for idx, score in ranked
        ]
