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
    frontend_base_url: str = "http://localhost:8501"

    llm_provider: LLMProviderName = "fake"
    llm_model: str = "nova-fake"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 1800

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

    secret_key: str = "dev-secret-change-me"
    jwt_algorithm: Literal["HS256"] = "HS256"
    jwt_expiration_minutes: int = 60 * 24 * 7
    auth_required: bool = False

    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None

    web_search_enabled: bool = False
    web_search_provider: Literal["tavily", "brave", "serpapi", "fake"] = "fake"
    web_search_api_key: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
