from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import get_settings
from app.core.cache_manager import CacheManager, MemoryCacheBackend, RedisCacheBackend
from app.core.erd_generator import ERDGenerator
from app.core.output_parser import OutputParser
from app.core.prompt_engine import PromptEngine
from app.core.quality_evaluator import QualityEvaluator
from app.core.retry_handler import RetryHandler
from app.core.schema_converter import SchemaConverter

if TYPE_CHECKING:
    from app.core.llm_router import LLMRouter


@lru_cache(maxsize=1)
def get_cache_manager() -> CacheManager:
    settings = get_settings()
    if settings.cache.backend == "redis":
        try:
            backend = RedisCacheBackend(settings.cache.redis_url)
        except Exception:
            backend = MemoryCacheBackend()
    else:
        backend = MemoryCacheBackend()
    return CacheManager(
        backend=backend,
        context_ttl=settings.cache.context_ttl,
        response_ttl=settings.cache.response_ttl,
    )


@lru_cache(maxsize=1)
def get_prompt_engine() -> PromptEngine:
    return PromptEngine(cache=get_cache_manager())


@lru_cache(maxsize=1)
def get_llm_router() -> "LLMRouter":
    from app.core.llm_router import LLMRouter
    return LLMRouter()


@lru_cache(maxsize=1)
def get_output_parser() -> OutputParser:
    return OutputParser()


@lru_cache(maxsize=1)
def get_schema_converter() -> SchemaConverter:
    return SchemaConverter()


@lru_cache(maxsize=1)
def get_erd_generator() -> ERDGenerator:
    return ERDGenerator()


@lru_cache(maxsize=1)
def get_quality_evaluator() -> QualityEvaluator:
    settings = get_settings()
    return QualityEvaluator(
        llm_router=get_llm_router(),
        prompt_engine=get_prompt_engine(),
        settings=settings,
    )


@lru_cache(maxsize=1)
def get_retry_handler() -> RetryHandler:
    settings = get_settings()
    return RetryHandler(
        prompt_engine=get_prompt_engine(),
        llm_router=get_llm_router(),
        output_parser=get_output_parser(),
        quality_evaluator=get_quality_evaluator(),
        max_retry=settings.quality.max_retry,
    )
