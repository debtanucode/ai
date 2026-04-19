"""Cosine similarity scorer using TF-IDF on token streams."""
from __future__ import annotations
import json
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseScorer


def _to_token_stream(obj: dict[str, Any]) -> str:
    """Serialize dict to a space-separated token string for TF-IDF."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    # keep all non-whitespace tokens
    import re
    tokens = re.findall(r"\S+", raw)
    return " ".join(tokens)


class CosineScorer(BaseScorer):
    """TF-IDF cosine similarity between JSON token streams."""

    def score(self, generated: dict[str, Any], golden: dict[str, Any]) -> float:
        gen_text = _to_token_stream(generated)
        gold_text = _to_token_stream(golden)
        if not gen_text.strip() and not gold_text.strip():
            return 1.0
        vectorizer = TfidfVectorizer(token_pattern=r"\S+")
        try:
            tfidf = vectorizer.fit_transform([gen_text, gold_text])
        except ValueError:
            return 0.0
        sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return float(max(0.0, min(1.0, sim)))
