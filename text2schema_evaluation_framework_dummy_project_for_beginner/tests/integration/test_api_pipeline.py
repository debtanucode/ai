"""Full API pipeline integration tests."""
from __future__ import annotations
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(test_client: AsyncClient):
    response = await test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ollama" in data


@pytest.mark.asyncio
async def test_evaluate_identical_json(test_client: AsyncClient):
    payload = {
        "generated": {"name": "Alice", "age": 30, "role": "admin"},
        "golden": {"name": "Alice", "age": 30, "role": "admin"},
    }
    response = await test_client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["composite_score"] > 0.7
    assert data["passed"] is True
    assert data["diff"]["matched"] == 3
    assert data["diff"]["missing"] == 0


@pytest.mark.asyncio
async def test_evaluate_completely_different(test_client: AsyncClient):
    payload = {
        "generated": {"x": 999, "y": "zzz"},
        "golden": {"a": 1, "b": "hello"},
    }
    response = await test_client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["composite_score"] < 0.7
    assert "run_id" in data


@pytest.mark.asyncio
async def test_evaluate_returns_run_id(test_client: AsyncClient):
    payload = {
        "generated": {"name": "Bob"},
        "golden": {"name": "Bob"},
        "run_id": "test-run-123",
    }
    response = await test_client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "test-run-123"


@pytest.mark.asyncio
async def test_evaluate_batch(test_client: AsyncClient):
    payload = [
        {
            "generated": {"a": 1},
            "golden": {"a": 1},
        },
        {
            "generated": {"b": 2},
            "golden": {"b": 9999},
        },
    ]
    response = await test_client.post("/api/evaluate/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2


@pytest.mark.asyncio
async def test_results_list_empty(test_client: AsyncClient):
    response = await test_client.get("/api/results")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_result_not_found(test_client: AsyncClient):
    response = await test_client.get("/api/results/nonexistent-run-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_invalid_weights_rejected(test_client: AsyncClient):
    payload = {
        "generated": {"a": 1},
        "golden": {"a": 1},
        "metric_config": {
            "jaccard": 0.5,
            "cosine": 0.5,
            "levenshtein": 0.5,
            "bleu": 0.5,
            "rouge": 0.5,
            "field_diff": 0.5,
            "llm_judge": 0.5,
        },
    }
    response = await test_client.post("/api/evaluate", json=payload)
    assert response.status_code == 422  # Pydantic validation error
