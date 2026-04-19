"""Unit tests for LLMJudgeScorer with mocked Ollama."""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from schemaeval.scorers.llm_judge import LLMJudgeScorer
from schemaeval.models.verdict import SemanticVerdict
import httpx


@pytest.fixture
def scorer():
    return LLMJudgeScorer(ollama_url="http://localhost:11434", model="test-model")


@pytest.mark.asyncio
async def test_successful_verdict(scorer):
    mock_response = {
        "response": json.dumps({
            "score": 0.9,
            "confidence": 0.85,
            "reasoning": "Very similar JSON structures."
        })
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        verdict = await scorer.score_with_verdict({"a": 1}, {"a": 1})

    assert verdict.llm_available is True
    assert verdict.score == pytest.approx(0.9)
    assert verdict.confidence == pytest.approx(0.85)
    assert "similar" in verdict.reasoning.lower()


@pytest.mark.asyncio
async def test_connect_error_returns_unavailable(scorer):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client_cls.return_value = mock_client

        verdict = await scorer.score_with_verdict({"a": 1}, {"b": 2})

    assert verdict.llm_available is False
    assert verdict.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_malformed_json_response(scorer):
    mock_response = {"response": "not valid json {{{"}
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        verdict = await scorer.score_with_verdict({"a": 1}, {"a": 1})

    assert verdict.llm_available is False


@pytest.mark.asyncio
async def test_score_clamped_to_bounds(scorer):
    mock_response = {
        "response": json.dumps({
            "score": 1.5,  # out of bounds
            "confidence": -0.1,  # out of bounds
            "reasoning": "test"
        })
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_client

        verdict = await scorer.score_with_verdict({"a": 1}, {"a": 1})

    assert 0.0 <= verdict.score <= 1.0
    assert 0.0 <= verdict.confidence <= 1.0
