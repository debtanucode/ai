"""CRUD operations for RunRecord and GoldenRecordRow."""
from __future__ import annotations
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..models.eval_result import EvalResult
from .models import RunRecord


class RunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, result: EvalResult) -> RunRecord:
        record = RunRecord(
            run_id=result.run_id,
            composite_score=result.composite_score,
            passed=result.passed,
            scores_json=result.scores.model_dump(),
            tags_json=result.tags,
            full_result_json=result.full_result_json(),
            created_at=result.created_at,
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return record

    async def get_by_id(self, run_id: str) -> RunRecord | None:
        result = await self._session.execute(
            select(RunRecord).where(RunRecord.run_id == run_id)
        )
        return result.scalar_one_or_none()

    async def list_recent(self, limit: int = 50) -> list[RunRecord]:
        result = await self._session.execute(
            select(RunRecord).order_by(RunRecord.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
