from __future__ import annotations

import json

import pytest

from app.core.schema_converter import SchemaConverter
from app.models.schema import OutputFormat, TargetDB


@pytest.fixture
def converter() -> SchemaConverter:
    return SchemaConverter()


def test_postgresql_ddl_contains_create_table(converter, sample_schema):
    ddl = converter.to_postgresql(sample_schema)
    assert "CREATE TABLE users" in ddl
    assert "CREATE TABLE orders" in ddl


def test_postgresql_ddl_contains_fk_constraint(converter, sample_schema):
    ddl = converter.to_postgresql(sample_schema)
    assert "FOREIGN KEY" in ddl
    assert "REFERENCES users" in ddl


def test_postgresql_ddl_contains_primary_key(converter, sample_schema):
    ddl = converter.to_postgresql(sample_schema)
    assert "PRIMARY KEY" in ddl


def test_postgresql_ddl_contains_index(converter, sample_schema):
    ddl = converter.to_postgresql(sample_schema)
    assert "CREATE UNIQUE INDEX" in ddl
    assert "idx_users_email" in ddl


def test_mysql_ddl_contains_engine(converter, sample_schema):
    from app.models.schema import SchemaDefinition, TargetDB
    mysql_schema = SchemaDefinition(
        tables=sample_schema.tables,
        target_db=TargetDB.mysql,
    )
    ddl = converter.to_mysql(mysql_schema)
    assert "ENGINE=InnoDB" in ddl
    assert "utf8mb4" in ddl


def test_mongodb_json_schema(converter, sample_schema):
    from app.models.schema import SchemaDefinition, TargetDB
    mongo_schema = SchemaDefinition(
        tables=sample_schema.tables,
        target_db=TargetDB.mongodb,
    )
    result = converter.to_mongodb_json_schema(mongo_schema)
    data = json.loads(result)
    assert "users" in data
    assert "$jsonSchema" in data["users"]


def test_dynamodb_json(converter, sample_schema):
    from app.models.schema import SchemaDefinition, TargetDB
    ddb_schema = SchemaDefinition(
        tables=sample_schema.tables,
        target_db=TargetDB.dynamodb,
    )
    result = converter.to_dynamodb(ddb_schema)
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) == 2
    table_names = [t["TableName"] for t in data]
    assert "users" in table_names
    assert "orders" in table_names


def test_convert_dispatches_correctly(converter, sample_schema):
    outputs = converter.convert(sample_schema, OutputFormat.sql)
    assert "sql" in outputs
    assert len(outputs["sql"]) > 0
