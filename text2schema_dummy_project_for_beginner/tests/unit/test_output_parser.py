from __future__ import annotations

import json

import pytest

from app.core.output_parser import OutputParser
from app.models.schema import SchemaDefinition


@pytest.fixture
def parser() -> OutputParser:
    return OutputParser()


def _make_schema_json(tables=None) -> str:
    if tables is None:
        tables = [
            {
                "name": "users",
                "columns": [
                    {"name": "id", "type": "BIGSERIAL", "primary_key": True, "nullable": False}
                ],
            }
        ]
    return json.dumps({"tables": tables, "target_db": "postgresql"})


def test_clean_json(parser):
    raw = _make_schema_json()
    schema, err = parser.parse(raw)
    assert schema is not None
    assert err is None
    assert isinstance(schema, SchemaDefinition)
    assert schema.tables[0].name == "users"


def test_fenced_json(parser):
    raw = "```json\n" + _make_schema_json() + "\n```"
    schema, err = parser.parse(raw)
    assert schema is not None
    assert err is None


def test_fenced_no_lang(parser):
    raw = "```\n" + _make_schema_json() + "\n```"
    schema, err = parser.parse(raw)
    assert schema is not None
    assert err is None


def test_json_with_prefix_text(parser):
    raw = "Here is the schema:\n\n" + _make_schema_json()
    schema, err = parser.parse(raw)
    assert schema is not None
    assert err is None


def test_invalid_json(parser):
    schema, err = parser.parse("this is not json at all")
    assert schema is None
    assert err is not None
    assert "JSON" in err or "extract" in err.lower()


def test_invalid_schema_shape(parser):
    raw = json.dumps({"bad_field": "value"})
    schema, err = parser.parse(raw)
    assert schema is None
    assert err is not None
    assert "validation" in err.lower() or "tables" in err.lower()


def test_empty_string(parser):
    schema, err = parser.parse("")
    assert schema is None
    assert err is not None


def test_returns_tuple_always(parser):
    result = parser.parse("garbage input !")
    assert isinstance(result, tuple)
    assert len(result) == 2
