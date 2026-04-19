"""LLM judge semantic verdict model."""
from pydantic import BaseModel


class SemanticVerdict(BaseModel):
    llm_available: bool = True
    score: float = 0.0          # 0.0–1.0
    confidence: float = 0.0     # 0.0–1.0
    reasoning: str = ""
    model_used: str = ""
