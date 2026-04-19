"""In-memory LRU cache keyed on SHA256 of the serialised request."""
from __future__ import annotations
import hashlib
import json
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Thread-safe in-memory LRU cache."""

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def make_key(data: dict[str, Any]) -> str:
        serialised = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)
