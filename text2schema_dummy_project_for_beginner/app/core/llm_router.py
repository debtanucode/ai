from __future__ import annotations

import asyncio
import random
from typing import Optional

from app.config import get_settings


class LLMRouter:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings.llm

        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError("langchain-ollama is required. Install with: pip install langchain-ollama") from e

        ollama_base_url = self._settings.ollama_base_url

        self._primary = ChatOllama(
            model=self._settings.primary_model,
            base_url=ollama_base_url,
            temperature=self._settings.temperature,
        )
        self._judge = ChatOllama(
            model=self._settings.judge_model,
            base_url=ollama_base_url,
            temperature=0.0,
        )

    async def _invoke_with_backoff(self, model, prompt: str, attempt: int = 0) -> str:
        from langchain_core.messages import HumanMessage
        delay = self._settings.backoff_base * (2 ** attempt) + random.uniform(0, 1)
        if attempt > 0:
            await asyncio.sleep(delay)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return response.content

    async def generate(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self._settings.max_retries):
            try:
                return await self._invoke_with_backoff(self._primary, prompt, attempt)
            except Exception as e:
                last_error = e
        raise RuntimeError(f"Ollama generate failed after {self._settings.max_retries} retries. Last error: {last_error}")

    async def judge(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self._settings.max_retries):
            try:
                return await self._invoke_with_backoff(self._judge, prompt, attempt)
            except Exception as e:
                last_error = e
        raise RuntimeError(f"Ollama judge failed after {self._settings.max_retries} retries. Last error: {last_error}")

    def get_available_providers(self) -> list[dict]:
        return [
            {
                "name": "ollama",
                "model": self._settings.primary_model,
                "role": "primary",
            }
        ]
