"""
api/config.py
-------------
Application configuration and environment variable management for FastAPI serving layer.
Defines paths, database connection parameters, security controls, and runtime modes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Model and Pipeline Artifact Paths
DEFAULT_MODEL_PATH = str(MODELS_DIR / "champion_model.joblib")
DEFAULT_PIPELINE_PATH = str(MODELS_DIR / "feature_pipeline.joblib")
DEFAULT_METRICS_PATH = str(MODELS_DIR / "model_metrics.json")
DEFAULT_DEMO_REPLAY_PATH = str(DATA_PROCESSED_DIR / "demo_replay_slice.json")


class Settings:
    """Application settings resolved from environment variables with safe defaults."""

    # Service Metadata
    APP_NAME: str = "Fraud Risk Analytics & Detection System API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")  # development, production, test
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # Host & Port
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Artifact Filepaths
    MODEL_PATH: str = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    PIPELINE_PATH: str = os.getenv("PIPELINE_PATH", DEFAULT_PIPELINE_PATH)
    METRICS_PATH: str = os.getenv("METRICS_PATH", DEFAULT_METRICS_PATH)
    DEMO_REPLAY_PATH: str = os.getenv("DEMO_REPLAY_PATH", DEFAULT_DEMO_REPLAY_PATH)

    # Database Configuration (Supabase PostgreSQL)
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")
    SUPABASE_DB_URL: str | None = os.getenv("SUPABASE_DB_URL")
    DB_POOL_MIN_CONN: int = int(os.getenv("DB_POOL_MIN_CONN", "1"))
    DB_POOL_MAX_CONN: int = int(os.getenv("DB_POOL_MAX_CONN", "10"))
    DB_CONNECT_TIMEOUT_SEC: int = int(os.getenv("DB_CONNECT_TIMEOUT_SEC", "5"))

    # Public API Security Controls
    # Allowed CORS Origins: comma-separated list of origins in env
    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000,https://*.vercel.app",
        ).split(",")
        if origin.strip()
    ]
    # Maximum allowed request body size in bytes (default 1 MB to prevent DoS)
    MAX_REQUEST_BODY_BYTES: int = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(1024 * 1024)))
    # Optional API Key guard (if set, requires X-API-Key header; if empty, allows open demo access)
    API_KEY: str | None = os.getenv("API_KEY", None)

    # Operational Decision Thresholds
    THRESHOLD_MEDIUM: float = float(os.getenv("THRESHOLD_MEDIUM", "0.10"))
    THRESHOLD_HIGH: float = float(os.getenv("THRESHOLD_HIGH", "0.35"))


settings = Settings()
