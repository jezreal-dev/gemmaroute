"""
config.py — Central configuration for GemmaRoute.

All settings are loaded from environment variables (or .env file).
The COST_PER_1M_TOKENS table and TIER_TO_MODEL mapping are derived from settings
and used by the graph nodes for cost calculation and routing logic.
"""
from typing import Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Ollama (local AMD ROCm or CPU fallback) ───────────────────────────────
    OLLAMA_HOST: str = "http://localhost:11434"
    LOCAL_ROUTER_MODEL: str = "gemma2:2b"      # Layer 1: classifier
    LOCAL_EXECUTOR_MODEL: str = "gemma:2b"     # Layer 2: simple-tier executor
    LOCAL_JUDGE_MODEL: str = "gemma:2b"        # Layer 3: quality judge

    # ── Fireworks AI ──────────────────────────────────────────────────────────
    FIREWORKS_API_KEY: str = ""
    FIREWORKS_BASE_URL: str = "https://api.fireworks.ai/inference/v1"
    CLOUD_MEDIUM_MODEL: str = "accounts/fireworks/models/deepseek-v4-flash"
    CLOUD_COMPLEX_MODEL: str = "accounts/fireworks/models/deepseek-v4-pro"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/app.db"
    
    # Security
    API_KEY: str = "gemmaroute-demo-2026"

    # ── Quality gate ──────────────────────────────────────────────────────────
    QUALITY_THRESHOLD: float = 0.75      # Minimum acceptable quality score
    MAX_ESCALATION_DEPTH: int = 2        # Max tier upgrades per request

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

# ── Cost table: USD per 1M tokens (blended input + output estimate) ───────────
COST_PER_1M_TOKENS: Dict[str, float] = {
    settings.LOCAL_ROUTER_MODEL:    0.00,   # local = always free
    settings.LOCAL_EXECUTOR_MODEL:  0.00,
    settings.LOCAL_JUDGE_MODEL:     0.00,
    settings.CLOUD_MEDIUM_MODEL:    0.20,   # Fireworks pricing estimate
    settings.CLOUD_COMPLEX_MODEL:   0.90,
}

# ── Tier ordering (escalation sequence) ───────────────────────────────────────
TIER_ORDER = ["trivial", "simple", "medium", "complex"]

# ── Maps tier name → model ID used for that tier ─────────────────────────────
TIER_TO_MODEL: Dict[str, str] = {
    "simple":  settings.LOCAL_EXECUTOR_MODEL,
    "medium":  settings.CLOUD_MEDIUM_MODEL,
    "complex": settings.CLOUD_COMPLEX_MODEL,
}
