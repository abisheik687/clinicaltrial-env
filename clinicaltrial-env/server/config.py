"""Environment configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    API_BASE_URL: str = "https://api.openai.com/v1"
    MODEL_NAME: str = "gpt-4o-mini"
    HF_TOKEN: str = ""
    ENV_URL: str = "http://localhost:7860"
    LOG_LEVEL: str = "INFO"
    SESSION_TIMEOUT_MINUTES: int = 30
    DEFAULT_SEED: int = 42


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()
