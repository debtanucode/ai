"""Field-level diff data models."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel


class FieldStatus(str, Enum):
    MATCHED = "matched"
    MISSING = "missing"    # field in golden but not in generated
    EXTRA = "extra"        # field in generated but not in golden
    MISMATCH = "mismatch"  # field present in both but values differ


class FieldDiff(BaseModel):
    path: str
    status: FieldStatus
    golden_value: Any = None
    generated_value: Any = None


class DiffResult(BaseModel):
    fields: list[FieldDiff] = []
    total_fields: int = 0
    matched: int = 0
    missing: int = 0
    extra: int = 0
    mismatched: int = 0

    @property
    def field_accuracy(self) -> float:
        if self.total_fields == 0:
            return 1.0
        return self.matched / self.total_fields
