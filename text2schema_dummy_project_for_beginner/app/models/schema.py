from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class TargetDB(str, Enum):
    postgresql = "postgresql"
    mysql = "mysql"
    mongodb = "mongodb"
    dynamodb = "dynamodb"


class OutputFormat(str, Enum):
    sql = "sql"
    nosql = "nosql"
    erd = "erd"
    all = "all"


class ForeignKeyDefinition(BaseModel):
    references_table: str
    references_column: str
    on_delete: str = "NO ACTION"
    on_update: str = "NO ACTION"


class ColumnDefinition(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[ForeignKeyDefinition] = None
    unique: bool = False
    index: bool = False
    default: Optional[str] = None
    comment: Optional[str] = None


class IndexDefinition(BaseModel):
    name: str
    columns: list[str]
    unique: bool = False
    index_type: str = "btree"


class ConstraintDefinition(BaseModel):
    name: str
    type: str
    expression: str


class TableDefinition(BaseModel):
    name: str
    columns: list[ColumnDefinition]
    indexes: list[IndexDefinition] = Field(default_factory=list)
    constraints: list[ConstraintDefinition] = Field(default_factory=list)
    comment: Optional[str] = None


class SchemaDefinition(BaseModel):
    tables: list[TableDefinition]
    target_db: TargetDB = TargetDB.postgresql
    version: str = "1.0"
    description: Optional[str] = None


class QualityScore(BaseModel):
    syntax: float = Field(ge=0.0, le=1.0)
    integrity: float = Field(ge=0.0, le=1.0)
    naming: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    composite: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = False

    # Hardcoded weights — do NOT import settings (circular import risk)
    _SYNTAX_W = 0.25
    _INTEGRITY_W = 0.25
    _NAMING_W = 0.15
    _COMPLETENESS_W = 0.35

    @model_validator(mode="after")
    def compute_composite(self) -> "QualityScore":
        self.composite = (
            self.syntax * self._SYNTAX_W
            + self.integrity * self._INTEGRITY_W
            + self.naming * self._NAMING_W
            + self.completeness * self._COMPLETENESS_W
        )
        self.passed = self.composite >= 0.8
        return self


class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class GenerateRequest(BaseModel):
    description: str = Field(min_length=10)
    target_db: TargetDB = TargetDB.postgresql
    output_format: OutputFormat = OutputFormat.sql
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    use_cache: bool = True


class GenerateResponse(BaseModel):
    schema_def: Optional[SchemaDefinition] = Field(default=None, alias="schema")
    quality: Optional[QualityScore] = None
    retry_count: int = 0
    outputs: dict[str, str] = Field(default_factory=dict)
    cached: bool = False
    processing_time_ms: float = 0.0

    model_config = {"populate_by_name": True}
