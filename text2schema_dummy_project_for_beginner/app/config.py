from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

_ROOT = Path(__file__).parent.parent


def _load_yaml() -> dict[str, Any]:
    yaml_path = _ROOT / "config.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    return {}


class LLMSettings(BaseSettings):
    primary: str = "ollama"
    primary_model: str = "llama3.2:3b"
    judge_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    fallback_order: list[str] = []
    temperature: float = 0.2
    max_tokens: int = 4096
    max_retries: int = 3
    backoff_base: float = 1.0


class QualitySettings(BaseSettings):
    threshold: float = 0.8
    weights: dict[str, float] = {
        "syntax": 0.25,
        "integrity": 0.25,
        "naming": 0.15,
        "completeness": 0.35,
    }
    max_retry: int = 3


class CacheSettings(BaseSettings):
    backend: str = "redis"
    context_ttl: int = 3600
    response_ttl: int = 86400
    redis_url: str = "redis://redis:6379/0"


class DatabaseSettings(BaseSettings):
    sandbox_dsn: str = "postgresql://sandbox:sandbox@postgres-sandbox:5432/sandbox"
    sandbox_schema_prefix: str = "sandbox_"


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]
    log_level: str = "info"


class Settings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    redis_url: str = ""
    sandbox_db_dsn: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def __init__(self, **data: Any) -> None:
        yaml_data = _load_yaml()
        # Merge YAML data as defaults, env vars will override via pydantic-settings
        merged: dict[str, Any] = {}
        if "llm" in yaml_data:
            merged["llm"] = LLMSettings(**yaml_data["llm"])
        if "quality" in yaml_data:
            merged["quality"] = QualitySettings(**yaml_data["quality"])
        if "cache" in yaml_data:
            merged["cache"] = CacheSettings(**yaml_data["cache"])
        if "database" in yaml_data:
            merged["database"] = DatabaseSettings(**yaml_data["database"])
        if "server" in yaml_data:
            merged["server"] = ServerSettings(**yaml_data["server"])
        merged.update(data)
        super().__init__(**merged)

    def model_post_init(self, __context: Any) -> None:
        # Allow top-level env vars to override nested settings
        if redis_url := os.getenv("REDIS_URL"):
            self.cache = CacheSettings(**{**self.cache.model_dump(), "redis_url": redis_url})
        if dsn := os.getenv("SANDBOX_DB_DSN"):
            self.database = DatabaseSettings(**{**self.database.model_dump(), "sandbox_dsn": dsn})
        if ollama_url := os.getenv("OLLAMA_BASE_URL"):
            self.llm = LLMSettings(**{**self.llm.model_dump(), "ollama_base_url": ollama_url})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
