"""Database migrations — create tables idempotently on startup."""
from .database import engine
from .models import Base


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
