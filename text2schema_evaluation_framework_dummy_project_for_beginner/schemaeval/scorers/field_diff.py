"""Field-level diff scorer with rich DiffResult output."""
from __future__ import annotations
from typing import Any
from ..models.diff import DiffResult, FieldDiff, FieldStatus
from .base import BaseScorer


def _flatten_dict(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested dict/list to {dot.path: value}."""
    result: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            result.update(_flatten_dict(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            result.update(_flatten_dict(v, path))
    else:
        result[prefix] = obj
    return result


class FieldDiffScorer(BaseScorer):
    """Compares fields between generated and golden JSON."""

    def score_with_details(
        self, generated: dict[str, Any], golden: dict[str, Any]
    ) -> DiffResult:
        gen_flat = _flatten_dict(generated)
        gold_flat = _flatten_dict(golden)

        all_paths = set(gen_flat.keys()) | set(gold_flat.keys())
        fields: list[FieldDiff] = []
        matched = missing = extra = mismatched = 0

        for path in sorted(all_paths):
            in_gen = path in gen_flat
            in_gold = path in gold_flat

            if in_gen and in_gold:
                if gen_flat[path] == gold_flat[path]:
                    status = FieldStatus.MATCHED
                    matched += 1
                else:
                    status = FieldStatus.MISMATCH
                    mismatched += 1
                fields.append(
                    FieldDiff(
                        path=path,
                        status=status,
                        golden_value=gold_flat[path],
                        generated_value=gen_flat[path],
                    )
                )
            elif in_gold and not in_gen:
                missing += 1
                fields.append(
                    FieldDiff(
                        path=path,
                        status=FieldStatus.MISSING,
                        golden_value=gold_flat[path],
                    )
                )
            else:
                extra += 1
                fields.append(
                    FieldDiff(
                        path=path,
                        status=FieldStatus.EXTRA,
                        generated_value=gen_flat[path],
                    )
                )

        total = len(gold_flat) + extra  # golden fields + extra generated fields
        return DiffResult(
            fields=fields,
            total_fields=total,
            matched=matched,
            missing=missing,
            extra=extra,
            mismatched=mismatched,
        )

    def score(self, generated: dict[str, Any], golden: dict[str, Any]) -> float:
        result = self.score_with_details(generated, golden)
        return result.field_accuracy
