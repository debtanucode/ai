"""Normalised Levenshtein similarity scorer."""
from __future__ import annotations
import json
from typing import Any
import Levenshtein as lev
from .base import BaseScorer


def _serialize(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class LevenshteinScorer(BaseScorer):
    """Normalised Levenshtein similarity (1 - normalised_distance)."""

    def score(self, generated: dict[str, Any], golden: dict[str, Any]) -> float:
        gen_str = _serialize(generated)
        gold_str = _serialize(golden)
        if not gen_str and not gold_str:
            return 1.0
        max_len = max(len(gen_str), len(gold_str))
        if max_len == 0:
            return 1.0
        distance = lev.distance(gen_str, gold_str)
        return float(max(0.0, 1.0 - distance / max_len))
