"""
config.py — Centralized configuration for DataPizza AI MVP.
All values are readable from environment variables; defaults are sensible for local dev.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PARSED_DIR: Path = DATA_DIR / "parsed"
DATABASE_DIR: Path = DATA_DIR / "database"
CACHE_DIR: Path = DATA_DIR / "cache"
OUTPUT_DIR: Path = ROOT_DIR / "output"
DATASET_DIR: Path = ROOT_DIR / "Dataset"
KB_DIR: Path = DATASET_DIR / "knowledge_base"
GROUND_TRUTH_DIR: Path = DATASET_DIR / "ground_truth"

# Ensure directories exist at import time
for _d in [RAW_DIR, PARSED_DIR, DATABASE_DIR, CACHE_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM / OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5.4-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_LOCATION: str = os.getenv(
    "QDRANT_LOCATION",
    str(DATABASE_DIR / "qdrant"),  # embedded / local path
)
QDRANT_HOST: str | None = os.getenv("QDRANT_HOST", None) or None
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None) or None

# Collection names
COLLECTION_MENU: str = "menu_index"
COLLECTION_MANUAL: str = "manual_index"
COLLECTION_CODE: str = "code_index"
COLLECTION_BLOG: str = "blog_index"
ALL_COLLECTIONS: list[str] = [
    COLLECTION_MENU,
    COLLECTION_MANUAL,
    COLLECTION_CODE,
    COLLECTION_BLOG,
]

# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------
FACTS_DB_PATH: Path = DATABASE_DIR / "facts.db"
INGESTION_LOG_DB_PATH: Path = DATABASE_DIR / "ingestion_log.db"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_MAX_CHAR_DEFAULT: int = int(os.getenv("CHUNK_MAX_CHAR_DEFAULT", "1000"))
CHUNK_MAX_CHAR_MENU: int = int(os.getenv("CHUNK_MAX_CHAR_MENU", "1000"))
CHUNK_MAX_CHAR_MANUAL: int = int(os.getenv("CHUNK_MAX_CHAR_MANUAL", "1200"))
CHUNK_MAX_CHAR_CODE: int = int(os.getenv("CHUNK_MAX_CHAR_CODE", "1200"))
CHUNK_MAX_CHAR_BLOG: int = int(os.getenv("CHUNK_MAX_CHAR_BLOG", "1000"))

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
DISH_MAPPING_PATH: Path = GROUND_TRUTH_DIR / "dish_mapping.json"
DOMANDE_CSV_PATH: Path = DATASET_DIR / "domande.csv"
SUBMISSION_CSV_PATH: Path = OUTPUT_DIR / "submission.csv"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
QDRANT_SEARCH_LIMIT: int = int(os.getenv("QDRANT_SEARCH_LIMIT", "10"))
QDRANT_SCORE_THRESHOLD: float = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.3"))

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
MAX_HANDOFFS_EASY: int = int(os.getenv("MAX_HANDOFFS_EASY", "2"))
MAX_HANDOFFS_MEDIUM: int = int(os.getenv("MAX_HANDOFFS_MEDIUM", "3"))
MAX_HANDOFFS_HARD: int = int(os.getenv("MAX_HANDOFFS_HARD", "5"))

# ---------------------------------------------------------------------------
# Tracing / OpenTelemetry
# ---------------------------------------------------------------------------
OTEL_ENDPOINT: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", None) or None
OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "datapizza-ai-mvp")
