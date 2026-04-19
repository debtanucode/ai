"""Abstract base class for all metric scorers."""
from abc import ABC, abstractmethod
from typing import Any


class BaseScorer(ABC):
    """All scorers implement score() returning a float in [0.0, 1.0]."""

    @abstractmethod
    def score(self, generated: dict[str, Any], golden: dict[str, Any]) -> float:
        """Compare generated JSON against golden reference. Returns 0.0â1.0."""
        ...
