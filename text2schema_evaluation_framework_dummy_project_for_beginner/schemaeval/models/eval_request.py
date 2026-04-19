"""Evaluation request and metric configuration models."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, model_validator


_DEFAULT_WEIGHT = round(1.0 / 7, 6)  # ≈ 0.142857


class MetricConfig(BaseModel):
    jaccard: float = _DEFAULT_WEIGHT
    cosine: float = _DEFAULT_WEIGHT
    levenshtein: float = _DEFAULT_WEIGHT
    bleu: float = _DEFAULT_WEIGHT
    rouge: float = _DEFAULT_WEIGHT
    field_diff: float = _DEFAULT_WEIGHT
    llm_judge: float = round(1.0 - _DEFAULT_WEIGHT * 6, 6)  # absorbs rounding

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "MetricConfig":
        total = (
            self.jaccard
            + self.cosine
            + self.levenshtein
            + self.bleu
            + self.rouge
            + self.field_diff
            + self.llm_judge
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Metric weights must sum to 1.0, got {total:.6f}")
        return self


class EvalRequest(BaseModel):
    generated: dict[str, Any]
    golden: dict[str, Any]
    metric_config: MetricConfig = MetricConfig()
    pass_threshold: float = 0.7
    run_id: str | None = None
    tags: list[str] = []
