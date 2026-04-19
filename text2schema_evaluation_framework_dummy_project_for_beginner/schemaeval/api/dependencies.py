"""FastAPI dependency factories."""
from __future__ import annotations
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from ..db.database import AsyncSessionLocal
from ..engine.evaluator import EvaluationEngine
from ..engine.cache import LRUCache
from ..config import settings

# Shared engine instance (created once at startup)
_shared_cache = LRUCache(max_size=settings.cache_max_size)
_shared_engine = EvaluationEngine(cache=_shared_cache)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def get_engine() -> EvaluationEngine:
    return _shared_engine
