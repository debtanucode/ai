"""Shared test fixtures."""
from __future__ import annotations
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from schemaeval.db.models import Base
from schemaeval.db.migrations import create_tables


@pytest.fixture
def simple_golden() -> dict:
    return {"name": "Alice", "age": 30, "role": "admin"}


@pytest.fixture
def simple_generated_identical() -> dict:
    return {"name": "Alice", "age": 30, "role": "admin"}


@pytest.fixture
def simple_generated_partial() -> dict:
    return {"name": "Alice", "age": 31, "role": "user"}


@pytest.fixture
def empty_dict() -> dict:
    return {}


@pytest.fixture
def nested_golden() -> dict:
    return {
        "user": {"name": "Bob", "email": "bob@example.com"},
        "scores": [1, 2, 3],
        "active": True,
    }


@pytest.fixture
def nested_generated_close() -> dict:
    return {
        "user": {"name": "Bob", "email": "bob@example.com"},
        "scores": [1, 2, 4],
        "active": True,
    }


@pytest_asyncio.fixture
async def in_memory_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(in_memory_engine):
    session_factory = async_sessionmaker(in_memory_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def test_client(in_memory_engine):
    """ASGI test client with in-memory SQLite database."""
    from schemaeval.api.app import create_app
    from schemaeval.api import dependencies

    # Patch the engine to use in-memory DB
    original_local = dependencies.AsyncSessionLocal
    session_factory = async_sessionmaker(in_memory_engine, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[dependencies.get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client

    app.dependency_overrides.clear()
