"""Results retrieval endpoints."""
from __future__ import annotations
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ...models.eval_result import EvalResult
from ...db.repository import RunRepository
from ..dependencies import get_db

router = APIRouter(prefix="/api", tags=["results"])


@router.get("/results", response_model=list[EvalResult])
async def list_results(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[EvalResult]:
    repo = RunRepository(db)
    records = await repo.list_recent(limit=limit)
    return [EvalResult.model_validate_json(r.full_result_json) for r in records]


@router.get("/results/{run_id}", response_model=EvalResult)
async def get_result(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvalResult:
    repo = RunRepository(db)
    record = await repo.get_by_id(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return EvalResult.model_validate_json(record.full_result_json)
