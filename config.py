from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


LLMProviderName = Literal[
    "openai-compatible",
    "openai",
    "anthropic",
    "gemini",
    "groq",
    "openrouter",
    "together",
    "ollama",
    "fake",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Nova Backend"
    environment: str = "development"
    cors_allowed_origins: str = "*"

    llm_provider: LLMProviderName = "fake"
    llm_model: str = "nova-fake"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 800

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    together_api_key: str | None = None

    database_url: str = "sqlite:///./nova.db"
    chroma_persist_path: str = "./chroma_db"
    bcl_collection_name: str = "nova_bcl"
    bcl_relevance_threshold: float = Field(default=0.8, ge=0.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
