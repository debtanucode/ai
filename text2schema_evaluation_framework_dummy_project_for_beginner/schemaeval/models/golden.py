"""Golden record model."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class GoldenRecord(BaseModel):
    id: str
    name: str
    description: str = ""
    schema_data: dict[str, Any]
    tags: list[str] = []
