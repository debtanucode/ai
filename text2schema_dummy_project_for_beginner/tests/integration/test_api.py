from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_llm_router,
    get_quality_evaluator,
    get_retry_handler,
)
from app.main import app
from app.models.schema import (
    ColumnDefinition,
    QualityScore,
    SchemaDefinition,
    TableDefinition,
    TargetDB,
)


def _make_valid_schema() -> SchemaDefinition:
    return SchemaDefinition(
        tables=[
            TableDefinition(
                name="users",
                columns=[
                    ColumnDefinition(name="id", type="BIGSERIAL", primary_key=True, nullable=False),
                    ColumnDefinition(name="email", type="VARCHAR(255)", nullable=False),
                ],
            )
        ],
        target_db=TargetDB.postgresql,
    )


def _make_passing_quality() -> QualityScore:
    return QualityScore(syntax=1.0, integrity=1.0, naming=1.0, completeness=0.9)


def _mock_llm_router():
    mock = MagicMock()
    mock.get_available_providers.return_value = [
        {"name": "anthropic", "model": "claude-3-5-sonnet-20241022", "role": "primary"}
    ]
    return mock


def _mock_quality_evaluator():
    mock = AsyncMock()
    mock.evaluate = AsyncMock(return_value=QualityScore(syntax=1.0, integrity=1.0, naming=1.0, completeness=0.9))
    return mock


def _mock_retry_handler_factory(schema, quality):
    mock = AsyncMock()
    mock.run = AsyncMock(return_value=(schema, quality, 0))
    return mock


@pytest.fixture(autouse=True)
def override_llm_deps():
    """Override LLM dependencies so tests work without langchain installed."""
    app.dependency_overrides[get_llm_router] = _mock_llm_router
    app.dependency_overrides[get_quality_evaluator] = _mock_quality_evaluator
    yield
    app.dependency_overrides.pop(get_llm_router, None)
    app.dependency_overrides.pop(get_quality_evaluator, None)
    app.dependency_overrides.pop(get_retry_handler, None)


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dialects(client):
    response = client.get("/api/dialects")
    assert response.status_code == 200
    dialects = response.json()
    assert isinstance(dialects, list)
    assert len(dialects) == 4
    ids = [d["id"] for d in dialects]
    assert "postgresql" in ids
    assert "mysql" in ids
    assert "mongodb" in ids
    assert "dynamodb" in ids


def test_providers(client):
    response = client.get("/api/providers")
    assert response.status_code == 200
    providers = response.json()
    assert isinstance(providers, list)
    assert len(providers) >= 1


def test_generate_with_mocked_retry_handler(client):
    valid_schema = _make_valid_schema()
    passing_quality = _make_passing_quality()

    app.dependency_overrides[get_retry_handler] = _mock_retry_handler_factory(valid_schema, passing_quality).__class__
    app.dependency_overrides[get_retry_handler] = lambda: _mock_retry_handler_factory(valid_schema, passing_quality)

    response = client.post(
        "/api/generate",
        json={
            "description": "A user management system with authentication",
            "target_db": "postgresql",
            "output_format": "sql",
            "use_cache": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "schema" in data
    assert "quality" in data
    assert data["quality"]["passed"] is True


def test_generate_description_too_short(client):
    # Override retry handler for this test too — FastAPI resolves deps before body validation
    app.dependency_overrides[get_retry_handler] = lambda: AsyncMock()
    response = client.post(
        "/api/generate",
        json={
            "description": "short",
            "target_db": "postgresql",
            "output_format": "sql",
        },
    )
    assert response.status_code == 422
