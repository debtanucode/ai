"""BLEU score scorer."""
from __future__ import annotations
import json
import re
from typing import Any
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .base import BaseScorer


def _tokenize(obj: dict[str, Any]) -> list[str]:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return re.findall(r"[\w.@\-]+", raw)


class BLEUScorer(BaseScorer):
    """Sentence BLEU with smoothing (method1) on JSON token sequences."""

    def score(self, generated: dict[str, Any], golden: dict[str, Any]) -> float:
        gen_tokens = _tokenize(generated)
        gold_tokens = _tokenize(golden)
        if not gold_tokens:
            return 1.0 if not gen_tokens else 0.0
        smoothie = SmoothingFunction().method1
        try:
            result = sentence_bleu([gold_tokens], gen_tokens, smoothing_function=smoothie)
        except Exception:
            result = 0.0
        return float(max(0.0, min(1.0, result)))
