from __future__ import annotations

import pytest

from app.core.erd_generator import ERDGenerator


@pytest.fixture
def generator() -> ERDGenerator:
    return ERDGenerator()


def test_node_count_matches_tables(generator, sample_schema):
    result = generator.generate(sample_schema)
    assert len(result["nodes"]) == len(sample_schema.tables)


def test_fk_edge_created(generator, sample_schema):
    result = generator.generate(sample_schema)
    edges = result["edges"]
    assert len(edges) >= 1
    # orders.user_id -> users.id
    fk_edge = next(
        (e for e in edges if e["source"] == "orders" and e["target"] == "users"), None
    )
    assert fk_edge is not None


def test_node_has_required_fields(generator, sample_schema):
    result = generator.generate(sample_schema)
    for node in result["nodes"]:
        assert "id" in node
        assert "type" in node
        assert node["type"] == "tableNode"
        assert "position" in node
        assert "x" in node["position"]
        assert "y" in node["position"]
        assert "data" in node
        assert "label" in node["data"]
        assert "columns" in node["data"]


def test_edge_has_sourcehandle_and_targethandle(generator, sample_schema):
    result = generator.generate(sample_schema)
    for edge in result["edges"]:
        assert "sourceHandle" in edge
        assert "targetHandle" in edge
        # Handles must be column names (not None)
        assert edge["sourceHandle"] is not None
        assert edge["targetHandle"] is not None


def test_edge_type_is_smoothstep(generator, sample_schema):
    result = generator.generate(sample_schema)
    for edge in result["edges"]:
        assert edge["type"] == "smoothstep"


def test_no_edges_when_no_fks(generator):
    from app.models.schema import ColumnDefinition, SchemaDefinition, TableDefinition, TargetDB
    schema = SchemaDefinition(
        tables=[
            TableDefinition(
                name="standalone",
                columns=[ColumnDefinition(name="id", type="BIGSERIAL", primary_key=True)],
            )
        ],
        target_db=TargetDB.postgresql,
    )
    result = generator.generate(schema)
    assert result["edges"] == []
