"""Central configuration. Everything is overridable via environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class LLMSettings:
    """LLM provider. 'none' = fully offline deterministic mode (extractive composer, rule-based planner)."""

    provider: str = field(default_factory=lambda: _env("MERIDIAN_LLM_PROVIDER", "none").lower())
    model: str = field(default_factory=lambda: _env("MERIDIAN_LLM_MODEL", ""))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_base_url: str = field(default_factory=lambda: _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    timeout_s: float = field(default_factory=lambda: _env_float("MERIDIAN_LLM_TIMEOUT", 60.0))
    max_retries: int = field(default_factory=lambda: _env_int("MERIDIAN_LLM_RETRIES", 3))
    max_tokens: int = field(default_factory=lambda: _env_int("MERIDIAN_LLM_MAX_TOKENS", 1200))
    agent_mode: str = field(default_factory=lambda: _env("MERIDIAN_AGENT_MODE", "auto").lower())  # auto|policy|llm_tools

    @property
    def effective_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == "anthropic":
            return "claude-sonnet-5-5"
        if self.provider == "openai":
            return "gpt-4o-mini"
        return ""


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("MERIDIAN_DATA_DIR", str(PROJECT_ROOT / "data"))))
    demo_dir: Path = field(default_factory=lambda: Path(_env("MERIDIAN_DEMO_DIR", str(PROJECT_ROOT / "demo_corpus"))))
    eval_dir: Path = field(default_factory=lambda: Path(_env("MERIDIAN_EVAL_DIR", str(PROJECT_ROOT / "eval_data"))))
    reports_dir: Path = field(default_factory=lambda: Path(_env("MERIDIAN_REPORTS_DIR", str(PROJECT_ROOT / "reports"))))
    host: str = field(default_factory=lambda: _env("MERIDIAN_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("MERIDIAN_PORT", 8000))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MERIDIAN_MAX_UPLOAD_MB", 60))
    render_dpi: int = field(default_factory=lambda: _env_int("MERIDIAN_RENDER_DPI", 170))
    ocr_lang: str = field(default_factory=lambda: _env("MERIDIAN_OCR_LANG", "eng"))
    # Chunking
    chunk_target_words: int = 150
    chunk_max_words: int = 230
    # Retrieval
    candidate_pool: int = 60
    final_k: int = 6
    rrf_k: int = 60
    # Agent
    max_retrieval_rounds: int = 3
    faithfulness_threshold: float = 0.6
    llm: LLMSettings = field(default_factory=LLMSettings)

    def ensure_dirs(self) -> None:
        for sub in ("", "pages", "figures", "uploads", "tables"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
