"""ROUGE-L scorer."""
from __future__ import annotations
import json
from typing import Any
from rouge_score import rouge_scorer
from .base import BaseScorer


def _serialize(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class ROUGEScorer(BaseScorer):
    """ROUGE-L F-measure on serialised JSON strings."""

    def __init__(self) -> None:
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def score(self, generated: dict[str, Any], golden: dict[str, Any]) -> float:
        if not generated and not golden:
            return 1.0
        gen_str = _serialize(generated)
        gold_str = _serialize(golden)
        if gen_str == gold_str:
            return 1.0
        scores = self._scorer.score(gold_str, gen_str)
        return float(max(0.0, min(1.0, scores["rougeL"].fmeasure)))
