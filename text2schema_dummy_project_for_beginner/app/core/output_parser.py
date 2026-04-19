from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import ValidationError

from app.models.schema import SchemaDefinition


class OutputParser:
    def parse(self, raw: str) -> tuple[Optional[SchemaDefinition], Optional[str]]:
        json_str = self._extract_json(raw)
        if json_str is None:
            return None, f"Could not extract JSON from LLM response. Raw output starts with: {raw[:200]!r}"
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}. Extracted string starts with: {json_str[:200]!r}"
        try:
            schema = SchemaDefinition.model_validate(data)
            return schema, None
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"])
                errors.append(f"  - {loc}: {err['msg']}")
            error_msg = "Schema validation failed:\n" + "\n".join(errors)
            return None, error_msg

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        # Strip markdown code fences
        fenced = re.sub(r"```(?:json)?\s*", "", text)
        fenced = re.sub(r"```\s*", "", fenced).strip()

        # Find outermost { ... }
        start = fenced.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(fenced)):
            if fenced[i] == "{":
                depth += 1
            elif fenced[i] == "}":
                depth -= 1
                if depth == 0:
                    return fenced[start : i + 1]
        return None
