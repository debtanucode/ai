from __future__ import annotations

from app.models.schema import SchemaDefinition


class ERDGenerator:
    _COL_WIDTH = 320
    _ROW_HEIGHT = 280
    _COLS = 3

    def generate(self, schema: SchemaDefinition) -> dict:
        nodes = []
        edges = []

        for i, table in enumerate(schema.tables):
            row = i // self._COLS
            col = i % self._COLS
            x = col * (self._COL_WIDTH + 40)
            y = row * (self._ROW_HEIGHT + 40)

            nodes.append({
                "id": table.name,
                "type": "tableNode",
                "position": {"x": x, "y": y},
                "data": {
                    "label": table.name,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.type,
                            "primary_key": c.primary_key,
                            "nullable": c.nullable,
                            "foreign_key": c.foreign_key.model_dump() if c.foreign_key else None,
                        }
                        for c in table.columns
                    ],
                },
            })

            for col in table.columns:
                if col.foreign_key:
                    fk = col.foreign_key
                    edge_id = f"e_{table.name}_{col.name}_{fk.references_table}"
                    edges.append({
                        "id": edge_id,
                        "source": table.name,
                        "sourceHandle": col.name,
                        "target": fk.references_table,
                        "targetHandle": fk.references_column,
                        "label": f"{col.name} → {fk.references_column}",
                        "type": "smoothstep",
                        "markerEnd": {"type": "arrowclosed"},
                    })

        return {"nodes": nodes, "edges": edges}
