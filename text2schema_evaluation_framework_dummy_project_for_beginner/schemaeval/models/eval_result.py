"""Evaluation result models."""
from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from .diff import DiffResult
from .verdict import SemanticVerdict


class MetricScores(BaseModel):
    jaccard: float = 0.0
    cosine: float = 0.0
    levenshtein: float = 0.0
    bleu: float = 0.0
    rouge: float = 0.0
    field_diff: float = 0.0
    llm_judge: float = 0.0


class EvalResult(BaseModel):
    run_id: str
    composite_score: float
    passed: bool
    scores: MetricScores
    diff: DiffResult
    verdict: SemanticVerdict
    generated: dict[str, Any]
    golden: dict[str, Any]
    tags: list[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def full_result_json(self) -> str:
        """Serialize to JSON string for DB blob storage."""
        return self.model_dump_json()


class BatchResult(BaseModel):
    results: list[EvalResult] = []
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_composite_score: float = 0.0
