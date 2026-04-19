from __future__ import annotations

import json
from typing import Optional

from app.models.schema import OutputFormat, SchemaDefinition, TargetDB


class SchemaConverter:
    def convert(self, schema: SchemaDefinition, output_format: OutputFormat) -> dict[str, str]:
        outputs: dict[str, str] = {}
        db = schema.target_db

        if output_format in (OutputFormat.sql, OutputFormat.all):
            if db == TargetDB.postgresql:
                outputs["sql"] = self.to_postgresql(schema)
            elif db == TargetDB.mysql:
                outputs["sql"] = self.to_mysql(schema)

        if output_format in (OutputFormat.nosql, OutputFormat.all):
            if db == TargetDB.mongodb:
                outputs["nosql"] = self.to_mongodb_json_schema(schema)
            elif db == TargetDB.dynamodb:
                outputs["nosql"] = self.to_dynamodb(schema)

        # If no output matched, produce default for db type
        if not outputs:
            if db in (TargetDB.postgresql, TargetDB.mysql):
                if db == TargetDB.postgresql:
                    outputs["sql"] = self.to_postgresql(schema)
                else:
                    outputs["sql"] = self.to_mysql(schema)
            else:
                if db == TargetDB.mongodb:
                    outputs["nosql"] = self.to_mongodb_json_schema(schema)
                else:
                    outputs["nosql"] = self.to_dynamodb(schema)

        return outputs

    def to_postgresql(self, schema: SchemaDefinition) -> str:
        lines: list[str] = []
        for table in schema.tables:
            if table.comment:
                lines.append(f"-- {table.comment}")
            lines.append(f"CREATE TABLE {table.name} (")
            col_defs: list[str] = []
            fk_constraints: list[str] = []

            for col in table.columns:
                parts = [f"    {col.name} {col.type}"]
                if col.primary_key:
                    parts.append("PRIMARY KEY")
                if not col.nullable and not col.primary_key:
                    parts.append("NOT NULL")
                if col.unique:
                    parts.append("UNIQUE")
                if col.default is not None:
                    parts.append(f"DEFAULT {col.default}")
                if col.comment:
                    parts.append(f"-- {col.comment}")
                col_defs.append(" ".join(parts))

                if col.foreign_key:
                    fk = col.foreign_key
                    constraint_name = f"fk_{table.name}_{col.name}"
                    fk_constraints.append(
                        f"    CONSTRAINT {constraint_name} FOREIGN KEY ({col.name}) "
                        f"REFERENCES {fk.references_table}({fk.references_column}) "
                        f"ON DELETE {fk.on_delete} ON UPDATE {fk.on_update}"
                    )

            for constraint in table.constraints:
                col_defs.append(f"    CONSTRAINT {constraint.name} {constraint.type} ({constraint.expression})")

            all_defs = col_defs + fk_constraints
            lines.append(",\n".join(all_defs))
            lines.append(");\n")

            for idx in table.indexes:
                unique_kw = "UNIQUE " if idx.unique else ""
                cols = ", ".join(idx.columns)
                using = f" USING {idx.index_type}" if idx.index_type != "btree" else ""
                lines.append(f"CREATE {unique_kw}INDEX {idx.name} ON {table.name} ({cols}){using};")

            lines.append("")

        return "\n".join(lines)

    def to_mysql(self, schema: SchemaDefinition) -> str:
        lines: list[str] = []
        for table in schema.tables:
            if table.comment:
                lines.append(f"-- {table.comment}")
            lines.append(f"CREATE TABLE `{table.name}` (")
            col_defs: list[str] = []
            fk_constraints: list[str] = []

            for col in table.columns:
                parts = [f"    `{col.name}` {col.type}"]
                if not col.nullable and not col.primary_key:
                    parts.append("NOT NULL")
                if col.primary_key:
                    parts.append("AUTO_INCREMENT PRIMARY KEY") if "INT" in col.type.upper() else parts.append("PRIMARY KEY")
                if col.unique and not col.primary_key:
                    parts.append("UNIQUE")
                if col.default is not None:
                    parts.append(f"DEFAULT {col.default}")
                if col.comment:
                    parts.append(f"COMMENT '{col.comment}'")
                col_defs.append(" ".join(parts))

                if col.foreign_key:
                    fk = col.foreign_key
                    constraint_name = f"fk_{table.name}_{col.name}"
                    fk_constraints.append(
                        f"    CONSTRAINT `{constraint_name}` FOREIGN KEY (`{col.name}`) "
                        f"REFERENCES `{fk.references_table}`(`{fk.references_column}`) "
                        f"ON DELETE {fk.on_delete} ON UPDATE {fk.on_update}"
                    )

            for idx in table.indexes:
                unique_kw = "UNIQUE " if idx.unique else ""
                cols = ", ".join(f"`{c}`" for c in idx.columns)
                col_defs.append(f"    {unique_kw}INDEX `{idx.name}` ({cols})")

            all_defs = col_defs + fk_constraints
            lines.append(",\n".join(all_defs))
            comment_clause = f" COMMENT='{table.comment}'" if table.comment else ""
            lines.append(f") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci{comment_clause};\n")

        return "\n".join(lines)

    def to_mongodb_json_schema(self, schema: SchemaDefinition) -> str:
        result: dict = {}
        for table in schema.tables:
            properties: dict = {}
            required: list[str] = []
            for col in table.columns:
                bson_type = self._pg_to_bson(col.type)
                properties[col.name] = {"bsonType": bson_type}
                if col.comment:
                    properties[col.name]["description"] = col.comment
                if not col.nullable:
                    required.append(col.name)
            validator: dict = {
                "bsonType": "object",
                "properties": properties,
            }
            if required:
                validator["required"] = required
            result[table.name] = {"$jsonSchema": validator}
        return json.dumps(result, indent=2)

    def to_dynamodb(self, schema: SchemaDefinition) -> str:
        tables = []
        for table in schema.tables:
            pk_cols = [c for c in table.columns if c.primary_key]
            attr_defs = []
            key_schema = []
            for i, col in enumerate(pk_cols[:2]):  # max 2 key attrs
                attr_defs.append({"AttributeName": col.name, "AttributeType": "S"})
                key_schema.append({"AttributeName": col.name, "KeyType": "HASH" if i == 0 else "RANGE"})
            if not attr_defs:
                attr_defs = [{"AttributeName": "PK", "AttributeType": "S"}]
                key_schema = [{"AttributeName": "PK", "KeyType": "HASH"}]
            tables.append({
                "TableName": table.name,
                "AttributeDefinitions": attr_defs,
                "KeySchema": key_schema,
                "BillingMode": "PAY_PER_REQUEST",
            })
        return json.dumps(tables, indent=2)

    @staticmethod
    def _pg_to_bson(pg_type: str) -> str:
        t = pg_type.upper()
        if any(x in t for x in ("INT", "SERIAL")):
            return "int"
        if any(x in t for x in ("FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "REAL")):
            return "double"
        if "BOOL" in t:
            return "bool"
        if any(x in t for x in ("DATE", "TIME", "TIMESTAMP")):
            return "date"
        if any(x in t for x in ("JSONB", "JSON")):
            return "object"
        if "UUID" in t:
            return "string"
        return "string"
