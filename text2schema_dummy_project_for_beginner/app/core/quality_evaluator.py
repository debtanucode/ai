from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.llm_router import LLMRouter
    from app.core.prompt_engine import PromptEngine

from app.models.schema import QualityScore, SchemaDefinition, TargetDB

_RESERVED_WORDS = {
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "asymmetric",
    "authorization", "between", "bigint", "binary", "both", "case", "cast", "check",
    "collate", "column", "constraint", "create", "cross", "current_date", "current_role",
    "current_time", "current_timestamp", "current_user", "default", "deferrable", "desc",
    "distinct", "do", "else", "end", "except", "false", "fetch", "for", "foreign",
    "from", "full", "grant", "group", "having", "in", "inner", "intersect", "into",
    "is", "join", "leading", "left", "like", "limit", "natural", "not", "null",
    "offset", "on", "only", "or", "order", "outer", "primary", "references",
    "returning", "right", "select", "session_user", "similar", "some", "symmetric",
    "table", "then", "to", "trailing", "true", "union", "unique", "user", "using",
    "when", "where", "window", "with",
}

_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class QualityEvaluator:
    def __init__(self, llm_router: "LLMRouter", prompt_engine: "PromptEngine", settings=None) -> None:
        self._llm = llm_router
        self._prompt_engine = prompt_engine
        self._settings = settings

    async def evaluate(self, schema: SchemaDefinition, original_description: str) -> QualityScore:
        syntax_score, _ = await self._score_syntax(schema)
        integrity_score, _ = self._score_integrity(schema)
        naming_score, _ = self._score_naming(schema)
        completeness_score, _ = await self._score_completeness(schema, original_description)

        return QualityScore(
            syntax=syntax_score,
            integrity=integrity_score,
            naming=naming_score,
            completeness=completeness_score,
        )

    async def _score_syntax(self, schema: SchemaDefinition) -> tuple[float, list[str]]:
        # For non-SQL targets, skip sandbox validation
        if schema.target_db not in (TargetDB.postgresql, TargetDB.mysql):
            return 1.0, []

        # Only attempt sandbox for PostgreSQL
        if schema.target_db != TargetDB.postgresql:
            return 1.0, []

        try:
            import asyncpg
        except ImportError:
            return 0.5, ["asyncpg not installed — skipping syntax check"]

        from app.config import get_settings
        from app.core.schema_converter import SchemaConverter

        settings = get_settings()
        dsn = settings.database.sandbox_dsn
        prefix = settings.database.sandbox_schema_prefix
        sandbox_name = f"{prefix}{uuid.uuid4().hex[:8]}"

        try:
            conn = await asyncpg.connect(dsn)
        except Exception as e:
            return 0.5, [f"Sandbox unreachable: {e}"]

        errors: list[str] = []
        try:
            await conn.execute(f"CREATE SCHEMA {sandbox_name}")
            await conn.execute(f"SET search_path TO {sandbox_name}")

            converter = SchemaConverter()
            ddl = converter.to_postgresql(schema)

            try:
                await conn.execute(ddl)
            except Exception as e:
                errors.append(str(e))

            return (1.0 if not errors else 0.0), errors
        finally:
            try:
                await conn.execute(f"DROP SCHEMA IF EXISTS {sandbox_name} CASCADE")
            except Exception:
                pass
            await conn.close()

    def _score_integrity(self, schema: SchemaDefinition) -> tuple[float, list[str]]:
        score = 1.0
        issues: list[str] = []
        table_names = {t.name for t in schema.tables}
        table_columns: dict[str, set[str]] = {
            t.name: {c.name for c in t.columns} for t in schema.tables
        }

        for table in schema.tables:
            # Check PK
            has_pk = any(c.primary_key for c in table.columns)
            if not has_pk:
                issues.append(f"Table '{table.name}' has no primary key")
                score = max(0.0, score - 0.1)

            # Check FKs
            for col in table.columns:
                if col.foreign_key:
                    fk = col.foreign_key
                    if fk.references_table not in table_names:
                        issues.append(
                            f"FK in {table.name}.{col.name} references unknown table '{fk.references_table}'"
                        )
                        score = max(0.0, score - 0.1)
                    elif fk.references_column not in table_columns.get(fk.references_table, set()):
                        issues.append(
                            f"FK in {table.name}.{col.name} references unknown column "
                            f"'{fk.references_table}.{fk.references_column}'"
                        )
                        score = max(0.0, score - 0.1)

            # Duplicate column names
            col_names = [c.name for c in table.columns]
            if len(col_names) != len(set(col_names)):
                issues.append(f"Table '{table.name}' has duplicate column names")
                score = max(0.0, score - 0.1)

        return score, issues

    def _score_naming(self, schema: SchemaDefinition) -> tuple[float, list[str]]:
        score = 1.0
        issues: list[str] = []

        for table in schema.tables:
            if not _SNAKE_RE.match(table.name):
                issues.append(f"Table name '{table.name}' is not snake_case")
                score = max(0.0, score - 0.05)
            if table.name.lower() in _RESERVED_WORDS:
                issues.append(f"Table name '{table.name}' is a reserved word")
                score = max(0.0, score - 0.05)
            if len(table.name) > 63:
                issues.append(f"Table name '{table.name}' exceeds 63 characters")
                score = max(0.0, score - 0.05)

            for col in table.columns:
                if not _SNAKE_RE.match(col.name):
                    issues.append(f"Column '{table.name}.{col.name}' is not snake_case")
                    score = max(0.0, score - 0.05)
                if col.name.lower() in _RESERVED_WORDS:
                    issues.append(f"Column '{table.name}.{col.name}' is a reserved word")
                    score = max(0.0, score - 0.05)
                if len(col.name) > 63:
                    issues.append(f"Column '{table.name}.{col.name}' exceeds 63 characters")
                    score = max(0.0, score - 0.05)

        return score, issues

    async def _score_completeness(
        self, schema: SchemaDefinition, description: str
    ) -> tuple[float, dict]:
        try:
            import json as _json

            schema_json = schema.model_dump_json(indent=2)
            prompt = await self._prompt_engine.build_judge_prompt(description, schema_json)
            raw = await self._llm.judge(prompt)

            # Extract JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                return 0.5, {"error": "No JSON in judge response"}

            data = _json.loads(raw[start : end + 1])
            entity_cov = float(data.get("entity_coverage", 0.5))
            rel_acc = float(data.get("relationship_accuracy", 0.5))
            attr_comp = float(data.get("attribute_completeness", 0.5))

            composite = entity_cov * 0.40 + rel_acc * 0.35 + attr_comp * 0.25
            return composite, data
        except Exception as e:
            return 0.5, {"error": str(e)}
