from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.output_parser import OutputParser
from app.core.retry_handler import RetryHandler
from app.models.schema import (
    ColumnDefinition,
    GenerateRequest,
    QualityScore,
    SchemaDefinition,
    TableDefinition,
    TargetDB,
)


def _make_valid_schema_json() -> str:
    schema = SchemaDefinition(
        tables=[
            TableDefinition(
                name="users",
                columns=[ColumnDefinition(name="id", type="BIGSERIAL", primary_key=True, nullable=False)],
            )
        ],
        target_db=TargetDB.postgresql,
    )
    return schema.model_dump_json()


@pytest.fixture
def passing_llm_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value=_make_valid_schema_json())
    return mock


@pytest.fixture
def passing_quality_evaluator_mock(passing_quality) -> AsyncMock:
    mock = AsyncMock()
    mock.evaluate = AsyncMock(return_value=passing_quality)
    return mock


@pytest.fixture
def failing_quality_evaluator_mock(failing_quality) -> AsyncMock:
    mock = AsyncMock()
    mock.evaluate = AsyncMock(return_value=failing_quality)
    return mock


async def test_success_on_first_attempt(
    passing_llm_mock, passing_quality_evaluator_mock, mock_cache
):
    from app.core.prompt_engine import PromptEngine
    prompt_engine = PromptEngine(cache=mock_cache)
    prompt_engine.build_generate_prompt = AsyncMock(return_value="test prompt")

    handler = RetryHandler(
        prompt_engine=prompt_engine,
        llm_router=passing_llm_mock,
        output_parser=OutputParser(),
        quality_evaluator=passing_quality_evaluator_mock,
        max_retry=3,
    )
    request = GenerateRequest(description="A simple user table", target_db=TargetDB.postgresql)
    schema, quality, retry_count = await handler.run(request)

    assert schema is not None
    assert quality is not None
    assert quality.passed is True
    assert retry_count == 0


async def test_retry_on_quality_failure(
    passing_llm_mock, failing_quality_evaluator_mock, mock_cache
):
    from app.core.prompt_engine import PromptEngine
    prompt_engine = PromptEngine(cache=mock_cache)
    prompt_engine.build_generate_prompt = AsyncMock(return_value="test prompt")

    handler = RetryHandler(
        prompt_engine=prompt_engine,
        llm_router=passing_llm_mock,
        output_parser=OutputParser(),
        quality_evaluator=failing_quality_evaluator_mock,
        max_retry=3,
    )
    request = GenerateRequest(description="A simple user table", target_db=TargetDB.postgresql)
    schema, quality, retry_count = await handler.run(request)

    assert schema is not None
    assert retry_count == 3  # exhausted max_retry
    # LLM was called max_retry+1 times
    assert passing_llm_mock.generate.call_count == 4
