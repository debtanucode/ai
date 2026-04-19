"""Application configuration via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    db_url: str = "sqlite+aiosqlite:///./schemaeval.db"

    # Ollama LLM Judge
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # API Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Caching
    cache_max_size: int = 256

    # Evaluation defaults
    default_pass_threshold: float = 0.7


settings = Settings()
