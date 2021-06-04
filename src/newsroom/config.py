"""Application settings loaded from environment variables and ``.env`` files.

All variables are prefixed with ``NEWSROOM_``; see ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProviderName = Literal["tfidf_svd", "sentence_transformers"]


class Settings(BaseSettings):
    """Runtime configuration for the CLI and the API.

    Values are read from the environment (``NEWSROOM_*``) and from a ``.env`` file in
    the working directory when present. Defaults keep everything fully offline.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEWSROOM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    artifacts_dir: Path = Field(default=Path("artifacts"), description="Model/index output dir")
    embedding_provider: EmbeddingProviderName = "tfidf_svd"
    embedding_dim: int = Field(default=256, ge=8, le=4096)
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    embedding_cache_dir: Path | None = None

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: str | None = Field(default=None, description="If set, requests must send the key")
    api_key_header: str = "X-API-Key"
    api_max_batch: int = Field(default=256, ge=1, le=10000)

    log_level: str = "INFO"
    random_seed: int = 42

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def _normalise_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Unsupported log level: {value!r}")
        return level

    @property
    def auth_enabled(self) -> bool:
        """Whether API-key authentication is active."""
        return bool(self.api_key)

    @property
    def classifier_path(self) -> Path:
        return self.artifacts_dir / "classifier.joblib"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings (useful in tests that mutate the environment)."""
    get_settings.cache_clear()
