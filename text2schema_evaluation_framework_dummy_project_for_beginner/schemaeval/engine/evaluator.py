"""Async evaluation engine orchestrating all 7 scorers."""
from __future__ import annotations
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config import settings
from ..models.eval_request import EvalRequest
from ..models.eval_result import EvalResult, MetricScores
from ..scorers.jaccard import JaccardScorer
from ..scorers.cosine import CosineScorer
from ..scorers.levenshtein import LevenshteinScorer
from ..scorers.bleu import BLEUScorer
from ..scorers.rouge import ROUGEScorer
from ..scorers.field_diff import FieldDiffScorer
from ..scorers.llm_judge import LLMJudgeScorer
from .cache import LRUCache

_executor = ThreadPoolExecutor(max_workers=4)


class EvaluationEngine:
    """Orchestrates all scorers and assembles EvalResult."""

    def __init__(self, cache: LRUCache | None = None) -> None:
        self._cache = cache or LRUCache(max_size=settings.cache_max_size)
        self._jaccard = JaccardScorer()
        self._cosine = CosineScorer()
        self._levenshtein = LevenshteinScorer()
        self._bleu = BLEUScorer()
        self._rouge = ROUGEScorer()
        self._field_diff = FieldDiffScorer()
        self._llm_judge = LLMJudgeScorer()

    async def evaluate(self, request: EvalRequest) -> EvalResult:
        cache_key = LRUCache.make_key(request.model_dump())
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        generated = request.generated
        golden = request.golden
        cfg = request.metric_config

        loop = asyncio.get_event_loop()

        # Run CPU-bound scorers in thread pool (parallel)
        def run_jaccard() -> float:
            return self._jaccard.score(generated, golden)

        def run_cosine() -> float:
            return self._cosine.score(generated, golden)

        def run_levenshtein() -> float:
            return self._levenshtein.score(generated, golden)

        def run_bleu() -> float:
            return self._bleu.score(generated, golden)

        def run_rouge() -> float:
            return self._rouge.score(generated, golden)

        def run_field_diff():
            return self._field_diff.score_with_details(generated, golden)

        (
            jaccard_score,
            cosine_score,
            levenshtein_score,
            bleu_score,
            rouge_score,
            diff_result,
            verdict,
        ) = await asyncio.gather(
            loop.run_in_executor(_executor, run_jaccard),
            loop.run_in_executor(_executor, run_cosine),
            loop.run_in_executor(_executor, run_levenshtein),
            loop.run_in_executor(_executor, run_bleu),
            loop.run_in_executor(_executor, run_rouge),
            loop.run_in_executor(_executor, run_field_diff),
            self._llm_judge.score_with_verdict(generated, golden),
        )

        field_diff_score = diff_result.field_accuracy
        llm_score = verdict.score if verdict.llm_available else 0.0

        # Compute composite score (weighted sum)
        composite = (
            cfg.jaccard * jaccard_score
            + cfg.cosine * cosine_score
            + cfg.levenshtein * levenshtein_score
            + cfg.bleu * bleu_score
            + cfg.rouge * rouge_score
            + cfg.field_diff * field_diff_score
            + cfg.llm_judge * llm_score
        )

        result = EvalResult(
            run_id=request.run_id or str(uuid.uuid4()),
            composite_score=round(composite, 6),
            passed=composite >= request.pass_threshold,
            scores=MetricScores(
                jaccard=round(jaccard_score, 6),
                cosine=round(cosine_score, 6),
                levenshtein=round(levenshtein_score, 6),
                bleu=round(bleu_score, 6),
                rouge=round(rouge_score, 6),
                field_diff=round(field_diff_score, 6),
                llm_judge=round(llm_score, 6),
            ),
            diff=diff_result,
            verdict=verdict,
            generated=generated,
            golden=golden,
            tags=request.tags,
        )
        self._cache.set(cache_key, result)
        return result
