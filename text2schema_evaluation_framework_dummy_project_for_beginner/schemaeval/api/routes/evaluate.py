"""Evaluation endpoints."""
from __future__ import annotations
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ...models.eval_request import EvalRequest
from ...models.eval_result import BatchResult, EvalResult
from ...engine.evaluator import EvaluationEngine
from ...db.repository import RunRepository
from ..dependencies import get_db, get_engine

router = APIRouter(prefix="/api", tags=["evaluate"])


async def _save_result(result: EvalResult, session: AsyncSession) -> None:
    repo = RunRepository(session)
    await repo.save(result)


@router.post("/evaluate", response_model=EvalResult)
async def evaluate(
    request: EvalRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    engine: EvaluationEngine = Depends(get_engine),
) -> EvalResult:
    result = await engine.evaluate(request)
    background_tasks.add_task(_save_result, result, db)
    return result


@router.post("/evaluate/batch", response_model=BatchResult)
async def evaluate_batch(
    requests: list[EvalRequest],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    engine: EvaluationEngine = Depends(get_engine),
) -> BatchResult:
    import asyncio

    results = await asyncio.gather(*[engine.evaluate(req) for req in requests])
    for result in results:
        background_tasks.add_task(_save_result, result, db)

    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.composite_score for r in results) / len(results) if results else 0.0

    return BatchResult(
        results=list(results),
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        avg_composite_score=round(avg_score, 6),
    )
