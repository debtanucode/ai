"""LLM Judge scorer via Ollama HTTP API."""
from __future__ import annotations
import json
from typing import Any
import httpx
from ..models.verdict import SemanticVerdict
from ..config import settings
from .base import BaseScorer

_PROMPT_TEMPLATE = """You are a JSON quality evaluator. Compare the generated JSON against the golden reference JSON and evaluate semantic similarity.

Golden reference JSON:
{golden}

Generated JSON:
{generated}

Evaluate how semantically similar the generated JSON is to the golden reference. Consider:
1. Are all required fields present?
2. Do the values match semantically (not just literally)?
3. Is the overall structure preserved?

Respond with a JSON object containing:
- "score": float between 0.0 and 1.0 (1.0 = perfect match)
- "confidence": float between 0.0 and 1.0
- "reasoning": brief explanation (1-2 sentences)
"""


class LLMJudgeScorer(BaseScorer):
    """Semantic scorer using Ollama LLM as judge."""

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._url = (ollama_url or settings.ollama_url).rstrip("/")
        self._model = model or settings.ollama_model

    async def score_with_verdict(
        self, generated: dict[str, Any], golden: dict[str, Any]
    ) -> SemanticVerdict:
        prompt = _PROMPT_TEMPLATE.format(
            golden=json.dumps(golden, indent=2),
            generated=json.dumps(generated, indent=2),
        )
        payload = {
            "model": self._model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self._url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                raw = json.loads(data.get("response", "{}"))
                score = float(raw.get("score", 0.5))
                confidence = float(raw.get("confidence", 0.5))
                reasoning = str(raw.get("reasoning", ""))
                return SemanticVerdict(
                    llm_available=True,
                    score=max(0.0, min(1.0, score)),
                    confidence=max(0.0, min(1.0, confidence)),
                    reasoning=reasoning,
                    model_used=self._model,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return SemanticVerdict(
                llm_available=False,
                score=0.0,
                confidence=0.0,
                reasoning="Ollama not available",
                model_used=self._model,
            )
        except Exception as exc:
            return SemanticVerdict(
                llm_available=False,
                score=0.0,
                confidence=0.0,
                reasoning=f"LLM judge error: {exc}",
                model_used=self._model,
            )

    def score(self, generated: dict[str, Any], golden: dict[str, Any]) -> float:
        """Sync wrapper â returns 0.0 when LLM unavailable (use score_with_verdict for async)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Canât run nested event loops; return 0.0 fallback
                return 0.0
            verdict = loop.run_until_complete(self.score_with_verdict(generated, golden))
            return verdict.score
        except Exception:
            return 0.0
