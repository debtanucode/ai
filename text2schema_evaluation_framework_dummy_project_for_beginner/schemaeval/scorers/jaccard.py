"""Jaccard similarity scorer â set overlap of flattened key=value pairs."""
from __future__ import annotations
import json
from typing import Any
from .base import BaseScorer


def _flatten(obj: Any, prefix: str = "") -> set[str]:
    """Recursively flatten a dict/list to a set of âdot.path=valueâ strings."""
    items: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            items |= _flatten(v, path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            items |= _flatten(v, path)
    else:
        items.add(f"{prefix}={json.dumps(obj, sort_keys=True)}")
    return items


class JaccardScorer(BaseScorer):
    """Jaccard similarity on flattened key=value string sets."""

    def score(self, generated: dict[str, Any], golden: dict[str, Any]) -> float:
        gen_set = _flatten(generated)
        gold_set = _flatten(golden)
        if not gen_set and not gold_set:
            return 1.0
        intersection = gen_set & gold_set
        union = gen_set | gold_set
        return len(intersection) / len(union)
