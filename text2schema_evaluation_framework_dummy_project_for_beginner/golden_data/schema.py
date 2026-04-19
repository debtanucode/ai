"""Load golden records from JSON files in samples/."""
from __future__ import annotations
import json
from pathlib import Path
from schemaeval.models.golden import GoldenRecord

_SAMPLES_DIR = Path(__file__).parent / "samples"


def load_golden_records(directory: Path | None = None) -> list[GoldenRecord]:
    """Load all golden records from the samples directory."""
    samples_dir = directory or _SAMPLES_DIR
    records: list[GoldenRecord] = []
    for path in sorted(samples_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            for item in raw:
                records.append(GoldenRecord.model_validate(item))
        else:
            records.append(GoldenRecord.model_validate(raw))
    return records
