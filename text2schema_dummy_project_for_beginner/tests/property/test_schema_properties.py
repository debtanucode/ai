from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.erd_generator import ERDGenerator
from app.core.output_parser import OutputParser
from app.core.schema_converter import SchemaConverter
from app.models.schema import (
    ColumnDefinition,
    SchemaDefinition,
    TableDefinition,
    TargetDB,
)


def column_strategy():
    return st.builds(
        ColumnDefinition,
        name=st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True),
        type=st.sampled_from(["BIGSERIAL", "VARCHAR(255)", "TEXT", "INTEGER", "BOOLEAN", "TIMESTAMPTZ"]),
        primary_key=st.booleans(),
        nullable=st.booleans(),
    )


def table_strategy():
    return st.builds(
        TableDefinition,
        name=st.from_regex(r"[a-z][a-z0-9_]{1,20}", fullmatch=True),
        columns=st.lists(column_strategy(), min_size=1, max_size=5),
    )


def schema_strategy():
    return st.builds(
        SchemaDefinition,
        tables=st.lists(table_strategy(), min_size=1, max_size=4),
        target_db=st.just(TargetDB.postgresql),
    )


@given(schema=schema_strategy())
@settings(max_examples=30)
def test_postgresql_always_has_create_table(schema):
    converter = SchemaConverter()
    ddl = converter.to_postgresql(schema)
    for table in schema.tables:
        assert f"CREATE TABLE {table.name}" in ddl


@given(schema=schema_strategy())
@settings(max_examples=30)
def test_erd_node_count_matches_tables(schema):
    generator = ERDGenerator()
    result = generator.generate(schema)
    assert len(result["nodes"]) == len(schema.tables)


@given(raw=st.text(min_size=0, max_size=500))
@settings(max_examples=50)
def test_output_parser_never_raises(raw):
    parser = OutputParser()
    result = parser.parse(raw)
    assert isinstance(result, tuple)
    assert len(result) == 2
    # Either (schema, None) or (None, error_str)
    schema, err = result
    assert schema is None or err is None
