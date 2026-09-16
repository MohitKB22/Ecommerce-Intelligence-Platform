"""Centralised application configuration.

All settings are environment-driven with production-safe defaults. The
application is *dual-mode*: it will use PostgreSQL/Redis when they are
reachable and transparently fall back to SQLite / in-process caching so the
stack is runnable with zero external services.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo_root/backend/app/core/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- application -------------------------------------------------
    APP_NAME: str = "E-Commerce Intelligence"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: Literal["development", "test", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    DEBUG: bool = False

    # ---- storage -----------------------------------------------------
    DATABASE_URL: str = f"sqlite:///{REPO_ROOT / 'data' / 'ecommerce.db'}"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False
    SQL_STATEMENT_TIMEOUT_MS: int = 15000

    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = True
    CACHE_DEFAULT_TTL: int = 300
    CACHE_MAX_ENTRIES: int = 20000

    VECTOR_DB_URL: str = ""  # empty => local numpy-backed index
    VECTOR_DIM: int = 128

    # ---- security ----------------------------------------------------
    SECRET_KEY: str = "dev-only-insecure-key-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12
    # Stored as a raw string, not list[str]: pydantic-settings JSON-decodes complex
    # types straight out of the dotenv file BEFORE any validator runs, so the
    # natural comma-separated value in .env raises SettingsError at import time.
    # Parsing happens in the `cors_origins` property instead.
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 300
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    ADMIN_EMAIL: str = "admin@ecommerce-intelligence.local"
    ADMIN_PASSWORD: str = "admin-change-me"

    # ---- ml ----------------------------------------------------------
    MODEL_PATH: str = str(REPO_ROOT / "ml" / "models")
    DATA_PATH: str = str(REPO_ROOT / "data")
    ML_AUTOLOAD: bool = True
    ML_DRIFT_THRESHOLD: float = 0.2

    # ---- workers -----------------------------------------------------
    CELERY_BROKER_URL: str = ""   # empty => in-process thread executor
    CELERY_RESULT_BACKEND: str = ""

    # ---- ranking / recommendation weights (configurable) -------------
    REC_W_COLLABORATIVE: float = 0.35
    REC_W_CONTENT: float = 0.25
    REC_W_POPULARITY: float = 0.15
    REC_W_PERSONALIZATION: float = 0.15
    REC_W_BUSINESS: float = 0.10

    RANK_W_TEXT: float = 0.30
    RANK_W_SEMANTIC: float = 0.25
    RANK_W_POPULARITY: float = 0.12
    RANK_W_RATING: float = 0.10
    RANK_W_CONVERSION: float = 0.10
    RANK_W_PERSONAL: float = 0.08
    RANK_W_AVAILABILITY: float = 0.05

    @field_validator("LOG_LEVEL")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    # ---- derived -----------------------------------------------------
    @property
    def cors_origins(self) -> list[str]:
        """Parsed allowlist. Accepts a comma-separated list or a JSON array."""
        raw = (self.CORS_ORIGINS or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            import json

            try:
                return [str(origin) for origin in json.loads(raw)]
            except (ValueError, TypeError):
                return []
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def model_dir(self) -> Path:
        p = Path(self.MODEL_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        p = Path(self.DATA_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def recommendation_weights(self) -> dict[str, float]:
        return {
            "collaborative": self.REC_W_COLLABORATIVE,
            "content": self.REC_W_CONTENT,
            "popularity": self.REC_W_POPULARITY,
            "personalization": self.REC_W_PERSONALIZATION,
            "business": self.REC_W_BUSINESS,
        }

    @property
    def ranking_weights(self) -> dict[str, float]:
        return {
            "text": self.RANK_W_TEXT,
            "semantic": self.RANK_W_SEMANTIC,
            "popularity": self.RANK_W_POPULARITY,
            "rating": self.RANK_W_RATING,
            "conversion": self.RANK_W_CONVERSION,
            "personal": self.RANK_W_PERSONAL,
            "availability": self.RANK_W_AVAILABILITY,
        }

    def validate_production(self) -> list[str]:
        """Return a list of production-readiness problems (empty == fine)."""
        problems: list[str] = []
        if not self.is_production:
            return problems
        if self.SECRET_KEY == "dev-only-insecure-key-change-me":
            problems.append("SECRET_KEY must be set to a strong random value in production")
        if self.ADMIN_PASSWORD == "admin-change-me":
            problems.append("ADMIN_PASSWORD must be changed in production")
        if self.is_sqlite:
            problems.append("SQLite is not supported in production; set DATABASE_URL to PostgreSQL")
        if self.DEBUG:
            problems.append("DEBUG must be disabled in production")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that mutate the environment."""
    get_settings.cache_clear()


settings = get_settings()

# Honour an explicit test flag so pytest never touches a developer database.
if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("ENVIRONMENT") is None:
    pass
