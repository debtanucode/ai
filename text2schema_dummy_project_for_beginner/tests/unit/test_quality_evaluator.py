from __future__ import annotations

import pytest

from app.models.schema import (
    ColumnDefinition,
    ForeignKeyDefinition,
    SchemaDefinition,
    TableDefinition,
    TargetDB,
)
from app.core.quality_evaluator import QualityEvaluator


def _make_evaluator(mock_llm, mock_cache):
    from app.core.prompt_engine import PromptEngine
    prompt_engine = PromptEngine(cache=mock_cache)
    return QualityEvaluator(llm_router=mock_llm, prompt_engine=prompt_engine)


def _single_table(columns) -> SchemaDefinition:
    return SchemaDefinition(
        tables=[TableDefinition(name="test_table", columns=columns)],
        target_db=TargetDB.postgresql,
    )


def test_missing_pk(mock_llm, mock_cache):
    evaluator = _make_evaluator(mock_llm, mock_cache)
    schema = _single_table([
        ColumnDefinition(name="email", type="VARCHAR(255)", nullable=False),
    ])
    score, issues = evaluator._score_integrity(schema)
    assert score < 1.0
    assert any("primary key" in i.lower() for i in issues)


def test_bad_fk_reference(mock_llm, mock_cache):
    evaluator = _make_evaluator(mock_llm, mock_cache)
    schema = SchemaDefinition(
        tables=[
            TableDefinition(
                name="orders",
                columns=[
                    ColumnDefinition(name="id", type="BIGSERIAL", primary_key=True),
                    ColumnDefinition(
                        name="user_id",
                        type="BIGINT",
                        foreign_key=ForeignKeyDefinition(
                            references_table="nonexistent_table",
                            references_column="id",
                            on_delete="NO ACTION",
                            on_update="NO ACTION",
                        ),
                    ),
                ],
            )
        ],
        target_db=TargetDB.postgresql,
    )
    score, issues = evaluator._score_integrity(schema)
    assert score < 1.0
    assert any("nonexistent_table" in i for i in issues)


def test_reserved_word_table_name(mock_llm, mock_cache):
    evaluator = _make_evaluator(mock_llm, mock_cache)
    schema = SchemaDefinition(
        tables=[
            TableDefinition(
                name="user",
                columns=[ColumnDefinition(name="id", type="BIGSERIAL", primary_key=True)],
            )
        ],
        target_db=TargetDB.postgresql,
    )
    score, issues = evaluator._score_naming(schema)
    assert score < 1.0
    assert any("reserved" in i.lower() for i in issues)


def test_camelcase_rejection(mock_llm, mock_cache):
    evaluator = _make_evaluator(mock_llm, mock_cache)
    schema = SchemaDefinition(
        tables=[
            TableDefinition(
                name="UserProfile",
                columns=[
                    ColumnDefinition(name="userId", type="BIGSERIAL", primary_key=True),
                ],
            )
        ],
        target_db=TargetDB.postgresql,
    )
    score, issues = evaluator._score_naming(schema)
    assert score < 1.0
    assert len(issues) >= 2  # Both table and column


def test_valid_schema_perfect_integrity(mock_llm, mock_cache, sample_schema):
    evaluator = _make_evaluator(mock_llm, mock_cache)
    score, issues = evaluator._score_integrity(sample_schema)
    assert score == 1.0
    assert issues == []


def test_valid_schema_perfect_naming(mock_llm, mock_cache, sample_schema):
    evaluator = _make_evaluator(mock_llm, mock_cache)
    score, issues = evaluator._score_naming(sample_schema)
    assert score == 1.0
    assert issues == []
