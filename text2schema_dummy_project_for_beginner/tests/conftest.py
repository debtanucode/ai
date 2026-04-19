from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.schema import (
    ColumnDefinition,
    ForeignKeyDefinition,
    GenerateRequest,
    IndexDefinition,
    QualityScore,
    SchemaDefinition,
    TableDefinition,
    TargetDB,
)


@pytest.fixture
def sample_schema() -> SchemaDefinition:
    users = TableDefinition(
        name="users",
        columns=[
            ColumnDefinition(name="id", type="BIGSERIAL", primary_key=True, nullable=False),
            ColumnDefinition(name="email", type="VARCHAR(255)", nullable=False, unique=True),
            ColumnDefinition(name="username", type="VARCHAR(100)", nullable=False),
            ColumnDefinition(name="created_at", type="TIMESTAMPTZ", nullable=False, default="NOW()"),
            ColumnDefinition(name="updated_at", type="TIMESTAMPTZ", nullable=False, default="NOW()"),
        ],
        indexes=[IndexDefinition(name="idx_users_email", columns=["email"], unique=True)],
    )
    orders = TableDefinition(
        name="orders",
        columns=[
            ColumnDefinition(name="id", type="BIGSERIAL", primary_key=True, nullable=False),
            ColumnDefinition(
                name="user_id",
                type="BIGINT",
                nullable=False,
                foreign_key=ForeignKeyDefinition(
                    references_table="users",
                    references_column="id",
                    on_delete="CASCADE",
                    on_update="NO ACTION",
                ),
            ),
            ColumnDefinition(name="total_amount", type="NUMERIC(10,2)", nullable=False),
            ColumnDefinition(name="status", type="VARCHAR(50)", nullable=False, default="'pending'"),
            ColumnDefinition(name="created_at", type="TIMESTAMPTZ", nullable=False, default="NOW()"),
        ],
    )
    return SchemaDefinition(tables=[users, orders], target_db=TargetDB.postgresql)


@pytest.fixture
def passing_quality() -> QualityScore:
    return QualityScore(syntax=1.0, integrity=1.0, naming=1.0, completeness=0.9)


@pytest.fixture
def failing_quality() -> QualityScore:
    return QualityScore(syntax=0.5, integrity=0.6, naming=0.7, completeness=0.5)


@pytest.fixture
def mock_llm() -> AsyncMock:
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value="{}")
    mock.judge = AsyncMock(
        return_value=json.dumps({
            "entity_coverage": 0.9,
            "relationship_accuracy": 0.9,
            "attribute_completeness": 0.9,
            "composite": 0.9,
            "reasoning": "Good schema",
            "missing_elements": [],
        })
    )
    return mock


@pytest.fixture
def mock_cache() -> AsyncMock:
    mock = AsyncMock()
    mock.get_context = AsyncMock(return_value=None)
    mock.set_context = AsyncMock(return_value=None)
    mock.get_response = AsyncMock(return_value=None)
    mock.set_response = AsyncMock(return_value=None)
    return mock
