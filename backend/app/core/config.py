"""
Application configuration – loaded from environment variables / .env file.
Uses pydantic-settings for validation and type coercion.
"""

import base64
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


def _infer_clerk_issuer_from_publishable_key(publishable_key: str) -> str:
    """Infer Clerk issuer URL from publishable key payload when possible."""
    key = (publishable_key or "").strip()
    if not key or not key.startswith("pk_"):
        return ""

    try:
        encoded = key.split("_", 2)[2]
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode((encoded + padding).encode("utf-8")).decode("utf-8", errors="ignore")
        host = decoded.strip().rstrip("$").strip().rstrip("/")
        if not host:
            return ""
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")
        return f"https://{host}"
    except Exception:
        return ""


class Settings(BaseSettings):
    """Centralised, validated app settings."""

    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "Lucida Lead Scoring API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # ── Database (Turso / libSQL) ────────────────────────────
    TURSO_DATABASE_URL: str = ""
    TURSO_AUTH_TOKEN: str = ""
    REQUIRE_TURSO_IN_PRODUCTION: bool = True
    SQLITE_DB_PATH: str = os.path.join(
        os.getenv("LOCALAPPDATA") or os.path.expanduser("~"),
        "Lucida",
        "lucida_local.db",
    )

    # ── Authentication (Clerk) ──────────────────────────────
    CLERK_SECRET_KEY: str = ""
    CLERK_PUBLISHABLE_KEY: str = ""
    CLERK_JWT_ISSUER: str = ""
    CLERK_JWT_AUDIENCE: str = ""
    CLERK_ALLOWED_AZP_ORIGINS: str = ""
    CLERK_JWT_LEEWAY_SECONDS: int = 10
    AUTH_BYPASS_ENABLED: bool = True

    # ── CORS ─────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:8000"
    TRUSTED_HOSTS: str = "localhost,127.0.0.1"
    ENABLE_API_DOCS: bool = True

    # ── ML ───────────────────────────────────────────────────
    MODEL_ARTIFACTS_DIR: str = "./model_artifacts"
    FREE_PLAN_MAX_MODELS: int = 1
    MAX_CSV_SIZE_MB: int = 200
    MAX_TOTAL_UPLOAD_SIZE_MB: int = 300
    MAX_UPLOAD_FILES: int = 8
    MAX_COLUMNS_PER_FILE: int = 500
    MAX_ROWS_PER_FILE: int = 2_000_000
    MAX_SCORE_RESPONSE_ROWS: int = 10_000
    SCORE_PERSISTENCE_MODE: str = "minimal"  # full | minimal | off
    MAX_PERSISTED_SCORED_ROWS_PER_RUN: int = 50_000
    SCORE_PERSISTENCE_TOP_N_FULL_PAYLOAD: int = 2_000
    SCORED_LEADS_RETENTION_DAYS: int = 14
    FEEDBACK_RETENTION_DAYS: int = 180
    FEEDBACK_LOOKUP_BATCH_SIZE: int = 500
    MAX_CONCURRENT_JOBS_PER_TENANT: int = 2
    RATE_LIMIT_TRAIN_PER_MIN: int = 8
    RATE_LIMIT_SCORE_PER_MIN: int = 30
    RATE_LIMIT_FEEDBACK_PER_MIN: int = 12
    UPLOAD_COMPRESSION_ENABLED: bool = True
    UPLOAD_COMPRESSION_MODE: str = "shadow"
    UPLOAD_COMPRESSION_NUMERIC_ONLY: bool = True
    UPLOAD_COMPRESSION_MIN_ROWS: int = 128
    UPLOAD_COMPRESSION_MAX_ALLOWED_MSE: float = 0.05
    UPLOAD_COMPRESSION_MAX_ALLOWED_IP_ERROR: float = 0.10

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def trusted_hosts_list(self) -> List[str]:
        """Parse comma-separated trusted hosts into a list."""
        return [host.strip() for host in self.TRUSTED_HOSTS.split(",") if host.strip()]

    @property
    def clerk_jwt_issuer(self) -> str:
        """Resolved Clerk JWT issuer, explicit setting first then inferred fallback."""
        explicit = self.CLERK_JWT_ISSUER.strip().rstrip("/")
        if explicit:
            return explicit
        return _infer_clerk_issuer_from_publishable_key(self.CLERK_PUBLISHABLE_KEY)

    @property
    def clerk_jwt_audience_list(self) -> List[str]:
        """Parse optional comma-separated JWT audience values."""
        return [aud.strip() for aud in self.CLERK_JWT_AUDIENCE.split(",") if aud.strip()]

    @property
    def clerk_allowed_azp_origins_list(self) -> List[str]:
        """Parse optional allowed authorized-party origins for Clerk tokens."""
        return [origin.strip().rstrip("/") for origin in self.CLERK_ALLOWED_AZP_ORIGINS.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    model_config = {
        "env_file": str(Path(__file__).resolve().parents[2] / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
